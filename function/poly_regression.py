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
        self.num_directions = num_directions
        self.num_steer_positions = num_steer_positions
        self.num_accel_positions = num_accel_positions
        
        self.episode_history_x = []
        self.episode_history_target = []
                
        self.num_inputs = 6 + 1
        
        # Hard-coded normalization for the RaceCar problem parameters
        shift_list = [0,
                     num_junctures / 2.,
                     num_lanes / 2.,
                     num_speeds / 2.,
                     num_directions / 2.,
                     num_steer_positions / 2.,
                     num_accel_positions / 2.]
        scale_list = [1,
                     num_junctures,
                     num_lanes,
                     num_speeds,
                     num_directions,
                     num_steer_positions,
                     num_accel_positions]

        if self.INCLUDE_SIN_COSINE:
            self.num_inputs += 4
            shift_list.extend([0,
                         0,
                         0,
                         0])
            scale_list.extend([1,
                         1,
                         1,
                         1])

        self.shift = np.asarray(shift_list)
        self.scale = np.asarray(scale_list)
        self._init_W()

        self.stat_error_cost = []
        self.stat_rel_error_cost = []
        self.stat_reg_cost = []
        self.stat_W = []
        
    def _init_W(self):
        self.W = np.random.randn(self.num_inputs ** self.POLY_DEGREE) * 10
        self.logger.debug("Initial W: %s", self.W)

    def _x_adjust(self, state, action):
        ''' Takes the input params and converts it to an input feature array
        '''
        x_list = [1, *state, *action]
        if self.INCLUDE_SIN_COSINE:
            # feature engineering
            juncture, direction = state[0], state[3]
            sj = math.sin(juncture * 2 * math.pi / self.num_junctures)
            cj = math.cos(juncture * 2 * math.pi / self.num_junctures)
            sd = math.sin(direction * 2 * math.pi / self.num_directions)
            cd = math.cos(direction * 2 * math.pi / self.num_directions)
            x_list.extend([sj, cj, sd, cd])
        
        x1 = np.asarray(x_list, dtype='float')
        x1 -= self.shift
        x1 /= self.scale

        x = x1
        if self.POLY_DEGREE >= 2:
            x = np.outer(x, x1).flatten()
            if self.POLY_DEGREE >= 3:
                x = np.outer(x, x1).flatten()

        return x
    
    def _value(self, X):
        # Calculate polynomial value X * W
        a = np.sum(X * self.W, axis=-1)
        return a
    
    def value(self, state, action):
        x = self._x_adjust(state, action)
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
        x = self._x_adjust(state, action)

        self.episode_history_x.append(x)
        self.episode_history_target.append(target)

    def update(self):
        ''' Updates the value function model based on data collected since
            the last update '''

        EHX = np.asarray(self.episode_history_x)
        EHT = np.asarray(self.episode_history_target)
        
        avg_target = np.mean(np.abs(EHT))
        self.logger.debug("Using divisor avg_target=%s", avg_target)
        N = len(EHT)
        
        period = max(10, self.max_iterations // 100) # for stats

        sum_error_cost = 0
        sum_rel_error_cost = 0
        sum_reg_cost = 0
        sum_W = 0
        debug_start_W = self.W.copy()
        
        # TODO: Stop upon convergence?
        for i in range(self.max_iterations):
            if self.batch_size == 0:
                # Do full-batch
                X = EHX   # N x d
                Y = EHT   # N
            else:
                ids = np.random.choice(N, size=self.batch_size)
                X = EHX[ids]   # b x d
                Y = EHT[ids]   # b
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
                self.logger.debug("Error: %0.2f \t dW:%0.2f\t W: %0.2f",
                                  error_cost, np.sum(dW**2), np.sum(self.W**2))

                self.stat_error_cost.append(sum_error_cost / period)
                self.stat_rel_error_cost.append(sum_rel_error_cost / period)
                self.stat_reg_cost.append(sum_reg_cost / period)
                self.stat_W.append(sum_W / period)
                sum_error_cost = 0
                sum_rel_error_cost = 0
                sum_reg_cost = 0
                sum_W = 0

        # Forget old episode history
        self.episode_history_x = []
        self.episode_history_target = []
        
        debug_diff_W = (debug_start_W - self.W)
        self.logger.debug(" W changed by %f, i.e.\n%s", 
                          np.sum(debug_diff_W ** 2), debug_diff_W)
        
    def collect_stats(self, ep):
        pass
    
    def report_stats(self, pref):
        self.logger.debug("Final W: %s", self.W)
        util.plot([self.stat_error_cost, self.stat_reg_cost],
                  range(len(self.stat_error_cost)),
                  ["Avg error cost", "Avg regularization cost"], pref="cost",
                  ylim=None)#(0, 50000))
        util.plot([self.stat_rel_error_cost], range(len(self.stat_rel_error_cost)),
                  ["Relative Avg error cost"], pref="relcost",
                  ylim=None)#(0, 50000))
        
        sW = np.asarray(self.stat_W).T
        util.plot(sW, range(sW.shape[1]), labels=None, pref="W", ylim=None)#(-10, 10))


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
    
    