'''
Created on Feb 9, 2019

@author: enerve
'''

import logging
import time
import util

import numpy as np
import torch

from function.value_function import ValueFunction

class MultiPolynomialRegression(ValueFunction):
    '''
    A function approximator that is a collection of action-specific
    polynomial regressions
    '''
    
    NUM_TEST_SAMPLES = 200
    
    def __init__(self,
                 alpha,
                 regularization_param,
                 batch_size,
                 max_iterations,
                 dampen_by,
                 feature_eng,
                 adam_update=True):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Using Multiple Polynomial Regression FA")
        
        self.alpha = alpha
        self.dampen_by = dampen_by
        self.regularization_param = regularization_param
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.feature_eng = feature_eng
        self.adam_update = adam_update
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')
        
        self.num_actions = feature_eng.num_actions()

        self.steps_history_sa = []
        self.steps_history_target = []
        self.is_updated = False
        
        self.W = feature_eng.initial_W()

        # for stats / debugging

        self.epoch_steps_history_shx = []
        self.epoch_steps_history_sht = []

        self.stat_error_cost = [[] for _ in range(self.num_actions)]
        self.stat_reg_cost = [[] for _ in range(self.num_actions)]
        self.stat_W = []
        self.sample_W_ids = np.random.choice(self.W.shape[0], size=30)
        self.stat_epoch_cost = []
        self.stat_epoch_cost_x = []
        
        self.iteration = 0
        
        if self.adam_update:
            self.first_moment = 0
            self.second_moment = 0

    def prefix(self):
        return 'multipoly_a%s_r%s_b%d_i%d_d%0.4f_F%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.dampen_by,
                                     self.feature_eng.prefix())


    def _value(self, X):
        # Calculate polynomial value X * W
        return X.matmul(self.W)
    
    def value(self, state, action):
        X = self.feature_eng.x_adjust(*state)
        ai = self.feature_eng.a_index(action)
        return self._value(X)[ai]

    def best_action(self, state):
        X = self.feature_eng.x_adjust(*state)
        V = self._value(X)
        
        i = torch.argmax(V).item()
        return self.feature_eng.a_tuple(i)

    def record(self, state, action, target):
        ''' Record incoming data for later training '''
        
        if self.is_updated:
            # Forget old steps history
            self.steps_history_sa = []
            self.steps_history_target = []
            self.is_updated = False

        l = len(self.steps_history_sa)
        
        if l <= 350000:
            self.steps_history_sa.append((*state, *action))
            self.steps_history_target.append(target)
            if l == 350000:
                self.logger.debug("========== Enough recorded! ==============")
        
    def store_training_data(self, fname):
        SHSA = np.asarray(self.steps_history_sa)
        SHT = np.asarray(self.steps_history_target)
        util.dump(SHSA, fname, "SA")
        util.dump(SHT, fname, "t")

    def load_training_data(self, fname, subdir):
        self.steps_history_sa = util.load(fname, subdir, suffix="SA").tolist()
        self.steps_history_target = util.load(fname, subdir, suffix="t").tolist()

    def update(self):
        ''' Updates the value function model based on data collected since
            the last update '''

        self.train()
        self.test()
        
        self.is_updated = True

    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def _prepare_data(self):
        steps_history_x = [[] for _ in range(self.num_actions)]
        steps_history_t = [[] for _ in range(self.num_actions)]
        self.logger.debug("Preparing training data for %d items",
                          len(self.steps_history_sa))
        for i, (sa, t) in enumerate(zip(self.steps_history_sa,
                         self.steps_history_target)):
            #if (i+1) < 250000:
            #    continue
            #if (i+1) >= 50000:
            #    break
            
            (j, l, s, d, st, ac) = sa
            ai = self.feature_eng.a_index((st, ac))
            x = self.feature_eng.x_adjust(j, l, s, d)
            steps_history_x[ai].append(x)
            steps_history_t[ai].append(t)
            if (i+1) % 10000 == 0:
                self.logger.debug("  prepared %d", i+1)
                
        return steps_history_x, steps_history_t

    def train(self):
        # Split up the training data by action
        steps_history_x, steps_history_t = self._prepare_data()

        # Train each action's dataset separately
        sWcollect = None
        epoch_shx = []
        epoch_sht = []
        before_unique = after_unique = 0
        for ai in range(self.num_actions):
            SHX = torch.stack(steps_history_x[ai]).to(self.device)
            SHT = torch.tensor(steps_history_t[ai]).to(self.device)
            a = self.feature_eng.a_tuple(ai)
            self.logger.debug("Train action %s", a)
            w, stat_err, stat_reg, stat_W  = self._train_action(
                SHX, SHT, self.W[:, ai])
            self.W[:, ai] = w
            self.stat_error_cost[ai].extend(stat_err)
            self.stat_reg_cost[ai].extend(stat_reg)
            
            # ---- Sample dataset for later testing/statistics
            SHX_test, SHT_test = SHX, SHT
            # Choose a smaller set before uniquifying
            ids = self._sample_ids(len(SHX_test), self.NUM_TEST_SAMPLES * 5)
            SHX_test, SHT_test = SHX_test[ids], SHT_test[ids]
            before_unique += SHX_test.shape[0]
            # Collect unique input-features
            # TODO: unique is kinda slow. Avoid it.
            dedup_SHX_test, ids = torch.unique(SHX_test, dim=0, return_inverse=True)
            if True:
                # To help debug how stable/consistent the dataset is,
                # plot how target values are distributed for a given input
                n_dedup_SHX_test = dedup_SHX_test.numpy()
                n_SHX_test = SHX_test.numpy()
                n_SHT_test = SHT_test.numpy()
                h = []
                for row in n_dedup_SHX_test:
                    match_ids = np.all(n_SHX_test == row, axis=1)
                    T = n_SHT_test[match_ids]
                    T -= T.mean()
                    h.extend(T.tolist())
                util.hist(h, 100, range=(-50,50), title="ai %s" % ai, pref="ai%d"%ai)
                #util.hist(h, 100, range=(0.5,1.5), title="ai %s" % ai, pref="ai%d"%ai)
            
            SHX_test = dedup_SHX_test
            SHT_test = SHT_test[ids]
            after_unique += SHX_test.shape[0]
            # Choose NUM_TEST_SAMPLES rows
            ids = self._sample_ids(len(SHX_test), self.NUM_TEST_SAMPLES)
            SHX_test, SHT_test = SHX_test[ids], SHT_test[ids]
            epoch_shx.append(SHX_test)
            epoch_sht.append(SHT_test)
            
            # stats
            # TODO: maybe run this on gpu
            sW = torch.stack(stat_W).cpu().numpy()  # n * mx
            sW = sW[:, self.sample_W_ids]
            sWcollect = sW if sWcollect is None else np.concatenate(
                [sWcollect, sW], axis=1) # n * (mx * ma)

            
        self.alpha *= (1 - self.dampen_by)

        self.stat_W.extend(list(sWcollect))
        self.logger.debug("Percentage of uniques: %d%%", 100 * after_unique / before_unique)

        # Save sampled dataset for future testing purposes
        self.epoch_steps_history_shx.append(epoch_shx)
        self.epoch_steps_history_sht.append(epoch_sht)
        
    def _train_action(self, SHX, SHT, init_W):

        W = init_W
        N = len(SHT)
        
        period = max(10, self.max_iterations // 100) # for stats
        if period > self.max_iterations:
            self.logger.warning("max_iterations too small for period plotting")

        sum_error_cost = 0
        sum_error_cost_trend = prev_sum_error_cost_trend = 0
        sum_reg_cost = 0
        sum_W = 0
        debug_start_W = W.clone().detach()
        
        stat_error_cost = []
        stat_reg_cost = []
        stat_W = []
        
        # TODO: decay alpha over the epochs too
        alpha = self.alpha
        
        for i in range(self.max_iterations):
            self.iteration += 1
            #self.logger.debug("  Iteration %d", i)
            if N > 0:
                if self.batch_size == 0:
                    # Do full-batch
                    X = SHX   # N x d
                    Y = SHT   # N
                else:
                    ids = self._sample_ids(N, self.batch_size)
                    X = SHX[ids]   # b x d
                    Y = SHT[ids]   # b
                V = X.matmul(W) # b
                D = V - Y # b
                
                # Calculate cost
                error_cost = 0.5 * torch.mean(D**2)
                reg_cost = 0.5 * self.regularization_param * torch.dot(W, W)
                
                # Find derivative
                DX = torch.unsqueeze(D, 1) * X  # b x d
                dW = -torch.mean(DX, dim=0)
                dW -= self.regularization_param * W
    
                
                # Update W
                
                if self.adam_update:
                    beta1 = 0.9
                    beta2 = 0.999
                    eps = 1e-8
                    
                    self.first_moment = beta1 * self.first_moment + (1-beta1) * dW
                    mt = self.first_moment / (1 - beta1 ** self.iteration)
                    self.second_moment = beta2 * self.second_moment + (1-beta2) * (dW ** 2)
                    vt = self.second_moment / (1 - beta2 ** self.iteration)
                
                    W += alpha * mt / (torch.sqrt(vt) + eps)
                else:
                    W += alpha * dW
            else:
                error_cost = 0
                reg_cost = 0

            #self.logger.debug("    Updated W by %0.2f",  np.sum(dW**2))
                         
            # Stats
            sum_error_cost += error_cost
            sum_reg_cost += reg_cost
            sum_W += W
            if (i+1) % period == 0:
                
                stat_error_cost.append(sum_error_cost / period)
                stat_reg_cost.append(sum_reg_cost / period)
                stat_W.append(sum_W / period)

#                 if (i+1) % (20*period) == 0:
#                     self.logger.debug("Error: %0.2f \t dW:%0.2f\t W: %0.2f",
#                                       error_cost, np.sum(dW**2),
#                                       np.sum(W**2))
#                     if prev_sum_error_cost_trend > 0:
#                         self.logger.debug("Progressive error cost: %0.2f",
#                                           sum_error_cost_trend / prev_sum_error_cost_trend)
#                         #if sum_error_cost_trend / prev_sum_error_cost_trend > 0.995:
#                         #    # Gradient descent has converged well
#                         #    break
#                     prev_sum_error_cost_trend = sum_error_cost_trend
#                     sum_error_cost_trend = 0
#                  
#                 sum_error_cost_trend += sum_error_cost / period

                sum_error_cost = 0
                sum_reg_cost = 0
                sum_W = 0

            alpha *= (1 - self.dampen_by)

        debug_diff_W = (debug_start_W - W)
        self.logger.debug("  trained \tN=%s \tW+=%0.2f \tE=%0.2f", N,
                          torch.dot(debug_diff_W, debug_diff_W),
                          stat_error_cost[-1])

        if False: 
            # Report error stats on full-batch, for debugging
            SA = torch.Tensor(self.steps_history_sa).to(self.device)
            X = SHX   # N x d
            Y = SHT   # N
     
            V = self._value(X) # b
            D = V - Y # b
             
            # Calculate cost
            error_cost = 0.5 * D**2
             
            for i in range(self.num_junctures):
                jid = SA[:, 0] == i
                if jid.any():
                    jerr = error_cost[jid].mean()
                    self.logger.debug("\t mean cost at juncture %2d : %0.2f",
                                      i, jerr)

#             import pandas as pd
#             dfX = pd.DataFrame(X.cpu().numpy())
#             dfY = pd.DataFrame(Y.cpu().numpy(), columns = ["target"])
#             df = pd.concat([dfX, dfY], axis=1)
#             df = df.groupby(["juncture", "lane"]).mean()
#             self.logger.debug("  Mean df: \n%s", df)
            
        return W, stat_error_cost, stat_reg_cost, stat_W
        
    def test(self):
        ''' Monitor the test performance of previously collected data 
        '''
        start_time = time.clock()
        overall_cost = 0
        num_tested = 0
        current_x = len(self.stat_error_cost[0])
        for e, (epoch_shx, epoch_sht) in enumerate(zip(
            self.epoch_steps_history_shx, self.epoch_steps_history_sht)):

            if e < len(self.epoch_steps_history_shx) - 25:
                # Discard outdated test data since it is no longer instructive
                self.epoch_steps_history_shx[e] = None
                self.epoch_steps_history_sht[e] = None
                continue
            
            #self.logger.debug("Testing epoch %d data", e)
            sum_cost = 0
            n = 0
            for ai in range(self.num_actions):
                SHX = epoch_shx[ai]
                SHT = epoch_sht[ai]
                error_cost, reg_cost = self._test_cost(
                    SHX, SHT, self.W[:, ai])
                sum_cost += error_cost
                n += len(SHT)

            if e >= len(self.stat_epoch_cost):
                # insert new epoch line, and prepopulate with zeros
                self.stat_epoch_cost.append([])
                self.stat_epoch_cost_x.append([])
            self.stat_epoch_cost[e].append(sum_cost / n)
            self.stat_epoch_cost_x[e].append(current_x-1)
            overall_cost += sum_cost
            num_tested += 1

        if num_tested > 0:
            self.logger.debug("Tested old data (%0.2f seconds) AvgError = %0.2fK",
                              time.clock() - start_time,
                              overall_cost / num_tested / 1000)
            
    def _test_cost(self, SHX, SHT, W):
        # Do full-batch
        X = SHX   # N x d
        Y = SHT   # N
        
        V = X.matmul(W) # N
        D = V - Y # N
        
        # Calculate cost
        error_cost = 0.5 * torch.dot(D, D)
        reg_cost = 0.5 * self.regularization_param * torch.dot(W, W)

        return error_cost, reg_cost

    def collect_stats(self, ep):
        pass
    
    def report_stats(self, pref=""):
        self.logger.debug("Final W: %s", self.W)
        util.plot(self.stat_error_cost,
                  range(len(self.stat_error_cost[0])),
                  labels = range(len(self.stat_error_cost)),
                  title = "MultiPoly training cost",
                  pref=pref+"cost",
                  ylim=None)#(0, 50000))
        util.plot_all(self.stat_epoch_cost,
                  self.stat_epoch_cost_x,
                  #labels = ['epoch %d data' % i for i in range(len(self.stat_epoch_cost))],
                  title = "MultiPoly error cost over recent epochs",
                  pref=pref+"old",
                  ylim=None)#(0, 50000))
        
        sW = np.asarray(self.stat_W).T
        util.plot(sW, range(sW.shape[1]), labels=None, pref="W", ylim=None)

    def save_model(self, pref=""):
        util.dump(self.W.cpu().numpy(), pref+"W")

    def load_model(self, load_subdir, pref=""):
        self.logger.debug("Loading W from: %s", load_subdir)
        self.W = util.load(pref+"W", load_subdir)
    