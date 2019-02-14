'''
Created on Nov 3, 2018

@author: enerve
'''

import logging
import math
import util

import numpy as np

from function.value_function import ValueFunction

class PolynomialRegression(ValueFunction):
    '''
    A function approximator that is a learned polynomial function.
    '''
    
    INCLUDE_SIN_COSINE = True
    SPLINE = True
    SPLINE_LENGTH = 4
    BOUNDED_FEATURES = True
    POLY_DEGREE = 3

    def __init__(self,
                 alpha,
                 regularization_param,
                 batch_size,
                 max_iterations,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions):
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

        self.num_junctures = num_junctures
        self.num_lanes = num_lanes
        self.num_speeds = num_speeds
        self.num_directions = num_directions
        self.num_steer_positions = num_steer_positions
        self.num_accel_positions = num_accel_positions
        
        # Collectors of incoming data
        self.steps_history_sa = []
        self.steps_history_target = []
                        
        # Hard-coded normalization for the RaceCar problem parameters
        
        shift_list = [0,
                     num_lanes / 2.,
                     num_speeds / 2.,
                     num_directions / 2.,
                     num_steer_positions / 2.,
                     num_accel_positions / 2.]
        scale_list = [1,
                     num_lanes,
                     num_speeds,
                     num_directions,
                     num_steer_positions,
                     num_accel_positions]

        if self.SPLINE:
            for knot in range(self.SPLINE_LENGTH, num_junctures,
                              self.SPLINE_LENGTH):
                shift_list.append(knot / 2)
                scale_list.append(knot)
        else:
            shift_list.append(num_junctures / 2.)
            scale_list.append(num_junctures)           

        if self.INCLUDE_SIN_COSINE:
            shift_list.extend([0,
                         0,
                         0,
                         0])
            scale_list.extend([1,
                         1,
                         1,
                         1])

        if self.BOUNDED_FEATURES:
            shift_list.extend([1,
                         1,
                         0.5,
                         0.5,
                         0.5])
            scale_list.extend([2,
                         2,
                         1,
                         1,
                         1])
            

        self.shift = np.asarray(shift_list)
        self.scale = np.asarray(scale_list)
        self.num_inputs = len(shift_list)
        self._init_W()

        # for stats / debugging

        self.stat_error_cost = []
        self.stat_rel_error_cost = []
        self.stat_reg_cost = []
        self.stat_W = []
        self.statQ = np.zeros((self.num_junctures,
                             self.num_lanes,
                             self.num_speeds,
                             self.num_directions,
                             self.num_steer_positions,
                             self.num_accel_positions))
        self.statC = np.zeros((self.num_junctures,
                             self.num_lanes,
                             self.num_speeds,
                             self.num_directions,
                             self.num_steer_positions,
                             self.num_accel_positions))

    def prefix(self):
        return 'poly_a%s_r%s_b%d_i%d_%d%s%s%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.POLY_DEGREE,
                                     't' if self.INCLUDE_SIN_COSINE else 'f',
                                     't' if self.SPLINE else 'f',
                                     't' if self.BOUNDED_FEATURES else 'f')

    def _init_W(self):
        self.W = np.random.randn(self.num_inputs ** self.POLY_DEGREE)

    def _x_adjust(self, juncture, lane, speed, direction, steer, accel):
        ''' Takes the input params and converts it to an input feature array
        '''
        x_list = [1, lane, speed, direction, steer, accel]

        if self.SPLINE:
            for knot in range(self.SPLINE_LENGTH, 28, self.SPLINE_LENGTH):
                x_list.append(max(0, -(juncture - knot)))
        else:
            x_list.append(juncture)

        if self.INCLUDE_SIN_COSINE:
            # feature engineering
            sj = math.sin(juncture * 2 * math.pi / self.num_junctures)
            cj = math.cos(juncture * 2 * math.pi / self.num_junctures)
            sd = math.sin(direction * 2 * math.pi / self.num_directions)
            cd = math.cos(direction * 2 * math.pi / self.num_directions)
            x_list.extend([sj, cj, sd, cd])
            
        if self.BOUNDED_FEATURES:
            x_list.append(max(0, lane - 1))
            x_list.append(max(0, 3 - lane))
            x_list.append(max(0, steer - 1))
            x_list.append(max(0, 1 - steer))
            x_list.append(max(0, accel - 1))

        x1 = np.asarray(x_list, dtype='float')
        x1 -= self.shift
        x1 /= self.scale

        x = x1
        if self.POLY_DEGREE >= 2:
            x = np.outer(x, x1).flatten()
            if self.POLY_DEGREE == 3:
                x = np.outer(x, x1).flatten()
            elif self.POLY_DEGREE == 4:
                x = np.outer(x, x).flatten()

        return x
    
    def _value(self, X):
        return np.dot(X, self.W)
    
    def value(self, state, action):
        x = self._x_adjust(*state, *action)
        return self._value(x)

    def best_action(self, S):
        best_v = float("-inf")
        best_action = None
        # TODO: Can this be vectorized instead?
        for steer in range(self.num_steer_positions):
            for accel in range(self.num_accel_positions):
                v = self.value(S, (steer, accel))
                if v > best_v:
                    best_v = v
                    best_action = (steer, accel)
        
        return best_action

    def record(self, state, action, target):
        ''' Record incoming data for later training '''
        self.steps_history_sa.append((*state, *action))
        self.steps_history_target.append(target)
        
        # For stats/debugging
        self.statQ[state + action] += target
        self.statC[state + action] += 1
        
    def store_training_data(self, fname):
        SHSA = np.asarray(self.steps_history_sa)
        SHT = np.asarray(self.steps_history_target)
        util.dump(SHSA, fname, "SA")
        util.dump(SHT, fname, "t")

    def load_training_data(self, fname, subdir):
        self.steps_history_sa = util.load(fname, subdir, suffix="SA")
        self.steps_history_target = util.load(fname, subdir, suffix="t")
        
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
        
        # Forget old steps history
        self.steps_history_sa = []
        self.steps_history_target = []

    def train(self):
        
        self.logger.debug("Preparing training data")
        steps_history_x = []
        for sa_items in self.steps_history_sa:
            x = self._x_adjust(*sa_items)
            steps_history_x.append(x)

        SHX = np.asarray(steps_history_x)
        SHT = np.asarray(self.steps_history_target)

        avg_target = np.mean(SHT)
        N = len(SHT)
        self.logger.debug("#training data N=%s", N)
        
        period = max(20, self.max_iterations // 100) # for stats

        sum_error_cost = 0
        sum_error_cost_trend = prev_sum_error_cost_trend = 0
        sum_rel_error_cost = 0
        sum_reg_cost = 0
        sum_W = 0
        debug_start_W = self.W.copy()
        
        for i in range(self.max_iterations):
            if self.batch_size == 0:
                # Do full-batch
                X = SHX   # N x d
                Y = SHT   # N
            else:
                ids = np.random.choice(N, size=self.batch_size)
                X = SHX[ids]   # b x d
                Y = SHT[ids]   # b
            V = self._value(X) # b
            D = V - Y # b
            
            # Calculate cost
            error_cost = 0.5 * np.mean(D**2)
            rel_error_cost = 0.5 * np.mean((D/avg_target)**2)
            reg_cost = 0.5 * self.regularization_param * np.sum(self.W ** 2)
            
            # Find derivative
            dW = -np.mean(D[:, np.newaxis] * X, axis=0)
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
        self.logger.debug(" W changed by %f", np.sum(debug_diff_W ** 2))

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
        
        sW = np.asarray(self.stat_W).T
        util.plot(sW, range(sW.shape[1]), labels=None, pref="W", ylim=None)#(-10, 10))

    def save_model(self, pref=""):
        util.dump(self.W, pref+"W")

    def load_model(self, load_subdir, pref=""):
        self.logger.debug("Loading W from: %s", load_subdir)
        self.W = util.load(pref+"W", load_subdir)

if __name__ == '__main__':
    alpha = 0.00001
    pr = PolynomialRegression(alpha, 0.2, 100, 2, 2, 2, 2, 3, 3)
    
    #target = 500
    stat = []
    for i in range(100000):
        s = tuple(10 * np.random.random(4))
        a = tuple(np.random.randint(0, 3, (2)))
        y = pr.value(s, a)
        #print("%s %s => %s" %(s, a, y))
        x_ = np.asarray([1, *s, *a])
        target = np.dot(x_, np.array([1,2,3,4,5,6,7]))
        
        delta = y - target
        stat.append(delta**2 + np.sum(pr.W ** 2))
        pr.update(s, a, alpha, delta)
        if i % 10000 == 0:
            print(i)
    print(pr.W)
    
    util.plot([stat], range(len(stat)), ['cost'])
    
    