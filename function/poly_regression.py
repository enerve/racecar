'''
Created on Nov 3, 2018

@author: enerve
'''

import util
from function.value_function import ValueFunction

import logging
import numpy as np
import torch

class PolynomialRegression(ValueFunction):
    '''
    A function approximator that is a learned polynomial function.
    '''
    
    MAX_TRAINING_SAMPLES = 100000
    
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

        self.logger.debug("Using Polynomial Regression FA")
        
        self.alpha = alpha
        self.regularization_param = regularization_param
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.feature_eng = feature_eng

        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        self.num_actions = feature_eng.num_actions()

        self.restart_record = True

        self.W = feature_eng.initial_W()
        self.train_X = []
        self.train_T = []
        self.train_count = 0

        # for stats / debugging

        self.stat_error_cost = []
        self.stat_rel_error_cost = []
        self.stat_reg_cost = []
        self.stat_W = []
        self.sample_W_ids = np.random.choice(self.W.shape[0], size=50)
#         self.statQ = np.zeros((self.num_junctures,
#                              self.num_lanes,
#                              self.num_speeds,
#                              self.num_directions,
#                              self.num_steer_positions,
#                              self.num_accel_positions))
#         self.statC = np.zeros((self.num_junctures,
#                              self.num_lanes,
#                              self.num_speeds,
#                              self.num_directions,
#                              self.num_steer_positions,
#                              self.num_accel_positions))

    def prefix(self):
        return 'poly_a%s_r%s_b%d_i%d_F%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.feature_eng.prefix())

    def _value(self, X):
        # Calculate polynomial value X * W
        with torch.no_grad():
            V = X.matmul(self.W)
        return V
    
    def value(self, state, action):
        x = self.feature_eng.x_adjust(*state, *action)
        return self._value(x)

    def best_action(self, S):
        best_v = float("-inf")
        best_action = None
        # TODO: Can this be vectorized instead?
        for a_index in range(self.num_actions):
            a_tuple = self.feature_eng.a_tuple(a_index)
            v = self.value(S, a_tuple)
            if v > best_v:
                best_v = v
                best_action = a_tuple
        
        return best_action

    def record(self, state, action, target):
        ''' Deprecated. Use add_data instead. 
            Record incoming data for later training '''

        self.logger.debug("Deprecated")
        if self.restart_record:
            # Forget old steps history
            self.steps_history_sa = []
            self.steps_history_target = []
            self.restart_record = False

        l = 0#len(self.steps_history_sa)
        
        if l <= 70000:
            self.steps_history_sa.append((*state, *action))
            self.steps_history_target.append(target)
            if l == 70000:
                self.logger.debug("========== Enough recorded! ==============")
        
        # For stats/debugging
        #self.statQ[state + action] += target
        #self.statC[state + action] += 1
        
    def add_data(self, x, target):
        if self.train_count == self.MAX_ROWS:
            self.logger.debug("Cannot accommodate more data")
            return
        
        if self.train_X is None:
            self.train_X = torch.zeros(self.MAX_ROWS, len(x)).to(self.device)
            self.train_T = torch.zeros(self.MAX_ROWS, 1).to(self.device)

        self.train_X[self.train_count] = x
        self.train_T[self.train_count] = target
        self.train_count += 1
        
    def store_training_data(self, fname):
        SHSA = np.asarray(self.steps_history_sa)
        SHT = np.asarray(self.steps_history_target)
        util.dump(SHSA, fname, "SA")
        util.dump(SHT, fname, "t")

    def load_training_data(self, fname, subdir):
        self.steps_history_sa = util.load(fname, subdir, suffix="SA").tolist()
        self.steps_history_target = util.load(fname, subdir, suffix="t").tolist()
        
#     def describe_training_data(self):
#         SHSA = np.asarray(self.steps_history_sa)
#         SHT = np.asarray(self.steps_history_target).reshape(130605, 1)
#         
#         SH = np.concatenate((SHSA, SHT), axis=1)
#         
#         import pandas as pd
#         df = pd.DataFrame(SH)
#         df = df.groupby(by=[0, 1, 2, 3, 4, 5]).max()
# 
# #         SH = df.values
#         SH = df.reset_index().values
#         self.steps_history_sa = SH[:, 0:6]
#         self.steps_history_target = SH[:, 6]
#         self.train()

    def update(self):
        ''' Updates the value function model based on data collected since
            the last update '''

        self.train()
        
        self.restart_record = True

    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def train(self):
        
        if self.steps_history_sa:
            self.logger.debug("Preparing training data")
            SHX = None
            N = len(self.steps_history_sa)
            for i, sa_items in enumerate(self.steps_history_sa):
                x = self.feature_eng.x_adjust(*sa_items)
                if SHX is None:
                    #TODO: preallocate a BIG but arbitrary-len SHX
                    SHX = torch.zeros(N, len(x)).to(self.device)
                SHX[i] = x
    
            SHT = torch.tensor(self.steps_history_target).to(self.device)
        else:
            self.logger_debug("Using collected training data")
            SHX, SHT = self.train_X, self.train_T
            N = self.train_count

        avg_target = torch.mean(SHT[0:N])
        self.logger.debug("#training data N=%s", N)
        
        period = max(20, self.max_iterations // 100) # for stats

        sum_error_cost = 0
        sum_error_cost_trend = prev_sum_error_cost_trend = 0
        sum_rel_error_cost = 0
        sum_reg_cost = 0
        sum_W = 0
        debug_start_W = self.W.clone()
        
        for i in range(self.max_iterations):
            if self.batch_size == 0:
                # Do full-batch
                X = SHX[0:N]   # N x d
                Y = SHT[0:N]   # N
            else:
                ids = self._sample_ids(N, self.batch_size)
                X = SHX[ids]   # b x d
                Y = SHT[ids]   # b
            V = X.matmul(self.W) # b
            D = V - Y # b
            
            # Calculate cost
            error_cost = 0.5 * torch.mean(D**2)
            rel_error_cost = 0.5 * torch.mean((D/avg_target)**2)
            reg_cost = 0.5 * self.regularization_param * torch.dot(self.W, self.W)
            
            # Find derivative
            DX = torch.unsqueeze(D, 1) * X  # b x d
            dW = -torch.mean(DX, dim=0)
            dW -= self.regularization_param * self.W
            
            # Update W
            self.W += self.alpha * dW
            
            # Stats
            sum_error_cost += error_cost
            sum_rel_error_cost += rel_error_cost
            sum_reg_cost += reg_cost
            sum_W += self.W
            if (i+1) % period == 0:
                
                self.stat_error_cost.append(sum_error_cost / period)
                self.stat_rel_error_cost.append(sum_rel_error_cost / period)
                self.stat_reg_cost.append(sum_reg_cost / period)
                self.stat_W.append(sum_W / period)

                if (i+1) % (20*period) == 0:
                    #self.logger.debug("Error: %0.2f \t dW:%0.2f\t W: %0.2f",
                    #                  error_cost, np.sum(dW**2),
                    #                  np.sum(self.W**2))
                    if prev_sum_error_cost_trend > 0:
                        self.logger.debug("Progressive error cost: %0.2f",
                                          sum_error_cost_trend / prev_sum_error_cost_trend)
                    #    if sum_error_cost_trend / prev_sum_error_cost_trend > 0.995:
                    #        # Gradient descent has converged well
                    #        break
                    prev_sum_error_cost_trend = sum_error_cost_trend
                    sum_error_cost_trend = 0
                
                sum_error_cost_trend += sum_error_cost / period

                sum_error_cost = 0
                sum_rel_error_cost = 0
                sum_reg_cost = 0
                sum_W = 0

        debug_diff_W = (debug_start_W - self.W)
        self.logger.debug(" W changed by %f",
                          torch.dot(debug_diff_W, debug_diff_W))

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
                    self.logger.debug("\t mean cost at juncture %2d : %0.2f", i, jerr)

        
    def plottable(self, axes, pick_max=False):
        ''' Returns a 2D plottable representation for debugging purposes.
            axes a tuple listing which dimensions of the six are to be flattened
            pick_max Whether to show the max or the sum along the flattened 
                dimensions
        '''

        with np.errstate(divide='ignore', invalid='ignore'):
            X = self.statQ / self.statC
        X = np.nanmax(X, axis=axes) if pick_max else np.nansum(X, axis=axes)
        return X.reshape(X.shape[0], -1).T

    def collect_stats(self, ep):
        pass
    
    def report_stats(self, pref=""):
        self.logger.debug("Final W: %s", self.W)
        util.plot([self.stat_error_cost, self.stat_reg_cost],
                  range(len(self.stat_error_cost)),
                  ["Avg error cost", "Avg regularization cost"], pref=pref+"cost",
                  ylim=None)#(0, 50000))
#         util.plot([self.stat_rel_error_cost], range(len(self.stat_rel_error_cost)),
#                   ["Relative Avg error cost"], pref=pref+"relcost",
#                   ylim=None)#(0, 50000))
        
        sW = torch.stack(self.stat_W).cpu().numpy()
        sW = sW[:, self.sample_W_ids].T
        util.plot(sW, range(sW.shape[1]), labels=None, pref="W", ylim=None)#(-10, 10))

    def save_model(self, pref=""):
        util.dump(self.W.cpu().numpy(), pref+"W")

    def load_model(self, load_subdir, pref=""):
        self.logger.debug("Loading W from: %s", load_subdir)
        self.W = util.load(pref+"W", load_subdir)
