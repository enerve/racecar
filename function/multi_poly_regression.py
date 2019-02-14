'''
Created on Feb 9, 2019

@author: enerve
'''

import logging
import time
import util

import numpy as np

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
                 feature_eng):
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
        self.num_actions = feature_eng.num_actions()

        self.steps_history_sa = []
        self.steps_history_target = []
        
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

    def prefix(self):
        return 'multipoly_a%s_r%s_b%d_i%d_d%0.4f_F%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.dampen_by,
                                     self.feature_eng.prefix())


    def _value(self, X):
        # Calculate polynomial value X * W
        return np.dot(X, self.W)
    
    def value(self, state, action):
        X = self.feature_eng.x_adjust(*state)
        ai = self.feature_eng.a_index(action)
        return self._value(X)[ai]

    def best_action(self, state):
        X = self.feature_eng.x_adjust(*state)
        V = self._value(X)
        
        i = np.argmax(V)
        return self.feature_eng.a_tuple(i)

    def record(self, state, action, target):
        ''' Record incoming data for later training '''
        self.steps_history_sa.append((*state, *action))
        self.steps_history_target.append(target)
        
    def store_training_data(self, fname):
        SHSA = np.asarray(self.steps_history_sa)
        SHT = np.asarray(self.steps_history_target)
        util.dump(SHSA, fname, "SA")
        util.dump(SHT, fname, "t")

    def load_training_data(self, fname, subdir):
        self.steps_history_sa = util.load(fname, subdir, suffix="SA")
        self.steps_history_target = util.load(fname, subdir, suffix="t")

    def update(self):
        ''' Updates the value function model based on data collected since
            the last update '''

        self.train()
        self.test()

    def train(self):
        # Split up the training data by action
        start_time = time.clock()
        steps_history_x = [[] for _ in range(self.num_actions)]
        steps_history_t = [[] for _ in range(self.num_actions)]
        for sa, t in zip(self.steps_history_sa,
                         self.steps_history_target):
            (j, l, s, d, st, ac) = sa
            ai = self.feature_eng.a_index((st, ac))
            x = self.feature_eng.x_adjust(j, l, s, d)
            steps_history_x[ai].append(x)
            steps_history_t[ai].append(t)
        self.logger.debug("Prepared training data (%0.2fs)",
                          time.clock() - start_time)

        # Train each action's dataset separately
        sWcollect = None
        epoch_shx = []
        epoch_sht = []
        for ai in range(self.num_actions):
            SHX = np.asarray(steps_history_x[ai])
            SHT = np.asarray(steps_history_t[ai])
            a = self.feature_eng.a_tuple(ai)
            self.logger.debug("Train action %s", a)
            w, stat_err, stat_reg, stat_W  = self._train_action(
                SHX, SHT, self.W[:, ai])
            self.W[:, ai] = w
            self.stat_error_cost[ai].extend(stat_err)
            self.stat_reg_cost[ai].extend(stat_reg)
            
            # Sample dataset for later testing
            SHX_test, u_ids = np.unique(SHX, axis=0, return_index=True)
            SHT_test = SHT[u_ids]
            ids = np.random.choice(len(SHX_test), size=self.NUM_TEST_SAMPLES)
            epoch_shx.append(SHX_test[ids])
            epoch_sht.append(SHT_test[ids])
            
            # stats
            sW = np.asarray(stat_W)  # n * mx
            sW = sW[:, self.sample_W_ids]
            sWcollect = sW if sWcollect is None else np.concatenate(
                [sWcollect, sW], axis=1) # n * (mx * ma)
            
        self.stat_W.extend(list(sWcollect))

        # Forget old steps history
        self.steps_history_sa = []
        self.steps_history_target = []

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
        debug_start_W = W.copy()
        
        stat_error_cost = []
        stat_reg_cost = []
        stat_W = []
        
        for i in range(self.max_iterations):
            if self.batch_size == 0:
                # Do full-batch
                X = SHX   # N x d
                Y = SHT   # N
            else:
                ids = np.random.choice(N, size=self.batch_size)
                X = SHX[ids]   # b x d
                Y = SHT[ids]   # b
            V = np.dot(X, W) # b
            D = V - Y # b
            
            # Calculate cost
            error_cost = 0.5 * np.mean(D**2)
            reg_cost = 0.5 * self.regularization_param * np.sum(W ** 2)
            
            # Find derivative
            dW = -np.mean(D[:, np.newaxis] * X, axis=0)
            dW -= self.regularization_param * W
            
            # Update W
            W += self.alpha * dW
            
            # Stats
            sum_error_cost += error_cost
            sum_reg_cost += reg_cost
            sum_W += W
            if (i+1) % period == 0:
                
                stat_error_cost.append(sum_error_cost / period)
                stat_reg_cost.append(sum_reg_cost / period)
                stat_W.append(sum_W / period)

#                 if (i+1) % (20*period) == 0:
#                     #self.logger.debug("Error: %0.2f \t dW:%0.2f\t W: %0.2f",
#                     #                  error_cost, np.sum(dW**2),
#                     #                  np.sum(W**2))
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

        debug_diff_W = (debug_start_W - W)
        self.logger.debug("  trained \tN=%s \tW+=%0.2f \tE=%0.2f", N,
                          np.sum(debug_diff_W ** 2),
                          stat_error_cost[-1])

        if False: 
            # Report error stats on full-batch, for debugging
            SA = np.asarray(self.steps_history_sa)
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

        self.alpha *= (1 - self.dampen_by)

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
        
        V = np.dot(X, W) # N
        D = V - Y # N
        
        # Calculate cost
        error_cost = 0.5 * np.sum(D**2)
        reg_cost = 0.5 * self.regularization_param * np.sum(W ** 2)

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
        util.dump(self.W, pref+"W")

    def load_model(self, load_subdir, pref=""):
        self.logger.debug("Loading W from: %s", load_subdir)
        self.W = util.load(pref+"W", load_subdir)
    