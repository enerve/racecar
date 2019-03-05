'''
Created on 1 Mar 2019

@author: enerve
'''

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import logging
from function.value_function import ValueFunction
from function.net import Net
import util

class NN_FA(ValueFunction):
    '''
    A neural-network action-value function approximator
    '''

    def __init__(self,
                 alpha,
                 regularization_param,
                 batch_size,
                 max_iterations,
                 feature_eng):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Using NN FA")

        self.alpha = alpha
        self.regularization_param = regularization_param
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.feature_eng = feature_eng

        # Collectors of incoming data
        self.steps_history_sa = []
        self.steps_history_target = []
        self.is_updated = False
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        self.num_outputs = feature_eng.num_actions()
        
        self.net = Net(4, 2000, self.num_outputs).cuda(self.device)
        self.criterion = nn.MSELoss(reduce=False)
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=alpha,
            weight_decay=regularization_param,
            momentum=0.9)

        # Stats / debugging
        self.stat_error_cost = []
        self.stat_reg_cost = []

    def prefix(self):
        return 'neural_a%s_r%s_b%d_i%d_F%s_NN%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.feature_eng.prefix(),
                                     self.net.prefix())

    def _value(self, state):
        X = self.feature_eng.x_adjust(*state)
        with torch.no_grad():
            output = self.net(X)
        return output
        
    def value(self, state, action):
        output = self._value(state)
        ai = self.feature_eng.a_index(action)
        return output[ai]

    def best_action(self, state):
        V = self._value(state)
        i = torch.argmax(V).item()
        return self.feature_eng.a_tuple(i)

    def record(self, state, action, target):
        ''' Record incoming data for later training '''
        if self.is_updated:
            # Forget old steps history
            self.steps_history_sa = []
            self.steps_history_target = []
            self.is_updated = False

        self.steps_history_sa.append((*state, *action))
        self.steps_history_target.append(target)

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

    def _prepare_data(self):
        steps_history_x = []
        steps_history_t = []
        steps_history_mask = []
        self.logger.debug("Preparing training data for %d items",
                          len(self.steps_history_sa))
        teye = torch.eye(self.num_outputs).to(self.device)
        for i, (sa, t) in enumerate(zip(self.steps_history_sa,
                         self.steps_history_target)):
            #if (i+1) < 250000:
            #    continue
            #if (i+1) >= 50000:
            #    break
            
            (j, l, s, d, st, ac) = sa
            ai = self.feature_eng.a_index((st, ac))
            x = self.feature_eng.x_adjust(j, l, s, d)
            m = teye[ai].clone()
            steps_history_x.append(x)
            steps_history_t.append(t)
            steps_history_mask.append(m)
            if (i+1) % 10000 == 0:
                self.logger.debug("  prepared %d", i+1)
                
        return steps_history_x, steps_history_t, steps_history_mask

    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def train(self):       
        steps_history_x, steps_history_t, steps_history_m = self._prepare_data()
        
        SHX = torch.stack(steps_history_x).to(self.device)
        SHT = torch.tensor(steps_history_t).to(self.device)
        SHM = torch.stack(steps_history_m).to(self.device)
        
        #W = self.W
        N = len(steps_history_t)
        
        # for stats
        preferred_samples = 100
        period = self.max_iterations // preferred_samples
        period = max(period, 10)
        if period > self.max_iterations:
            self.logger.warning("max_iterations too small for period plotting")

        sum_error_cost = torch.zeros(self.num_outputs).to(self.device)
        sum_error_cost.detach()
        sum_error_cost_trend = prev_sum_error_cost_trend = 0
        sum_reg_cost = torch.zeros(self.num_outputs).to(self.device)
#         sum_W = 0
#         debug_start_W = W.clone().detach()
        
        stat_error_cost = []
        stat_reg_cost = []
        
        
        for i in range(self.max_iterations):
            self.optimizer.zero_grad()
            
            if self.batch_size == 0:
                # Do full-batch
                X = SHX   # N x di
                Y = SHT   # N
                M = SHM   # N x do
            else:
                ids = self._sample_ids(N, self.batch_size)
                X = SHX[ids]   # b x di
                Y = SHT[ids]   # b
                M = SHM[ids]   # b x do
            Y = torch.unsqueeze(Y, 1)   # b x 1
            
            # forward
            outputs = self.net(X)       # b x do
            # loss
            Y = Y * M  # b x do
            loss = self.criterion(outputs, Y)  # b x do
            with torch.no_grad():
                # Zero-out the computed losses for the other actions/outputs
                loss *= M   # b x do
            # backward
            onez = torch.ones(loss.shape).to(self.device)
            loss.backward(onez)
            # loss.backward(M)
            
            # updated weights
            self.optimizer.step()
            
            # Stats
            sum_error_cost.add_(torch.mean(loss.detach(), 0))  # do
            #sum_W += W
            if (i+1) % period == 0:
                stat_error_cost.append(sum_error_cost.detach().cpu().numpy() / period)
                stat_reg_cost.append(sum_reg_cost.detach().cpu().numpy() / period)
                #stat_W.append(sum_W / period)

                #self.logger.debug("  loss=%0.2f", sum_error_cost.mean().item())

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

                torch.zeros(self.num_outputs, out=sum_error_cost)
                torch.zeros(self.num_outputs, out=sum_reg_cost)
                sum_W = 0

        self.logger.debug("  trained \tN=%s \tE=%0.2f", N,
                          stat_error_cost[-1].mean().item())

        self.stat_error_cost.extend(stat_error_cost)
        self.stat_reg_cost.extend(stat_reg_cost)
        
#         # ---- Sample dataset for later testing/statistics
#         SHX_test, SHT_test = SHX, SHT
#         # Choose a smaller set before uniquifying
#         ids = self._sample_ids(len(SHX_test), self.NUM_TEST_SAMPLES * 5)
#         SHX_test, SHT_test = SHX_test[ids], SHT_test[ids]
#         before_unique += SHX_test.shape[0]
#         # Collect unique input-features
#         # TODO: unique is kinda slow. Avoid it.
#         SHX_test, ids = torch.unique(SHX_test, dim=0, return_inverse=True)
#         SHT_test = SHT_test[ids]
#         after_unique += SHX_test.shape[0]
#         # Choose NUM_TEST_SAMPLES rows
#         ids = self._sample_ids(len(SHX_test), self.NUM_TEST_SAMPLES)
#         SHX_test, SHT_test = SHX_test[ids], SHT_test[ids]
#         epoch_shx.append(SHX_test)
#         epoch_sht.append(SHT_test)
        

    def test(self):
        pass

    def collect_stats(self, ep):
        pass
    
    def report_stats(self, pref=""):
        n_cost = np.asarray(self.stat_error_cost).T
        util.plot(n_cost,
                  range(n_cost.shape[1]),
                  labels = range(n_cost.shape[1]),
                  title = "NN training cost",
                  pref=pref+"cost",
                  ylim=None)
    
    def live_stats(self):
        n_cost = np.asarray(self.stat_error_cost).T
        util.plot(n_cost,
                  range(n_cost.shape[1]),
                  labels = range(n_cost.shape[1]),
                  title = "NN training cost",
                  live=True)

    def save_model(self, pref=""):
        pass

    def load_model(self, load_subdir, pref=""):
        pass