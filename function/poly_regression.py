'''
Created on Nov 3, 2018

@author: enerve
'''

import logging
import numpy as np

from function.value_function import ValueFunction

class PolynomialRegression(ValueFunction):
    '''
    A function approximator that is a trained polynomial function.
    '''

    def __init__(self,
                 alpha,
                 regularization_param,
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

        self.num_steer_positions = num_steer_positions
        self.num_accel_positions = num_accel_positions
        
        D = 6 + 1
        self.W = np.random.randn(D, D)
        self.logger.debug("Initial W: %s", self.W)
        
        self.shift = np.asarray([0,
                                 num_junctures / 2.,
                                 num_lanes / 2.,
                                 num_speeds / 2.,
                                 num_directions / 2.,
                                 num_steer_positions / 2.,
                                 num_accel_positions / 2.])
        self.scale = np.asarray([1, 
                                 num_junctures,
                                 num_lanes,
                                 num_speeds,
                                 num_directions,
                                 num_steer_positions,
                                 num_accel_positions])
        
#         self.X_sum = np.zeros(D, D)
#         self.count = 0
#         self.X_max = np.ones(D, D)
#         self.X_min = -np.ones(D, D)
#         self.shift = np.zeros(D, D)
        
    def value(self, state, action):
        x = np.asarray([1, *state, *action], dtype='float')
        # hard-coded and imperfect normalization
        x -= self.shift
        x /- self.scale
        a = np.dot(x, np.dot(self.W, x))
#         X = np.outer(x, x)
#         b = np.sum(X * self.W)
#         if a != b:
#             self.logger.debug(a/b)
        return a
    
    def best_action(self, S):
        best_v = float("-inf")
        best_action = None
        for steer in range(self.num_steer_positions):
            for accel in range(self.num_accel_positions):
                v = self.value(S, (steer, accel))
                if v > best_v:
                    best_v = v
                    best_action = (steer, accel)
        
        return best_action

    def update(self, state, action, delta):
        x = np.asarray([1, *state, *action], dtype='float')
        # hard-coded and imperfect normalization
        x -= self.shift
        x /= self.scale
        
        X = np.outer(x, x)
        
        #Normalization
#         self.X_sum += X
#         self.count += 1
#         self.X_max = np.maximum(self.X_max, X)
#         self.X_min = np.minimum(self.X_max, X)
#         new_scale = self.X_max - self.X_min
#         new_shift = self.X_sum / self.count
#         with np.errstate(divide='ignore'):
#             self.W *= self.scale / new_scale
#             self.W += self.shift - new_shift
#             self.scale = new_scale
#             self.shift = new_shift
#             X -= self.shift
#             X /= self.scale
        
        error_cost = 0.5 * delta**2
        reg_cost = 0.5 * self.regularization_param * np.sum(self.W ** 2)
        
        dW = - delta * X - self.regularization_param * self.W
        #dW[0][0] += self.regularization_param * self.W[0][0]
        
        self.W += self.alpha * dW
        
        return error_cost, reg_cost, np.sum(self.W), self.W.flatten()
        #return error_cost, reg_cost

if __name__ == '__main__':
    pr = PolynomialRegression(2, 2, 2, 2, 3, 3)
    
    alpha = 0.00001
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
    
    import util
    
    util.plot([stat], range(len(stat)), ['cost'])
    
    