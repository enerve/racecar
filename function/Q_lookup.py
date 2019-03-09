'''
Created on Nov 3, 2018

@author: enerve
'''

import logging
import numpy as np
import util
from . import ValueFunction

class QLookup(ValueFunction):
    '''
    A function approximator that stores value as table-lookups.
    '''

    def __init__(self,
                 config,
                 alpha):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Q is the learned value of a state/action
        self.Q = np.zeros((config.NUM_JUNCTURES,
                           config.NUM_LANES,
                            config.NUM_SPEEDS,
                            config.NUM_DIRECTIONS,
                            config.NUM_STEER_POSITIONS,
                            config.NUM_ACCEL_POSITIONS))
        
        self.alpha = alpha
        #self.episode_history = []

        self.stat_error_cost = []
        self.error_running_avg = 0
        
    def prefix(self):
        return 'Qtable_a%s_' % (self.alpha)

    def value(self, state, action):
        return self.Q[state + action]

    def best_action(self, S):
        As = self.Q[S]
        steer, accel = np.unravel_index(np.argmax(As, axis=None), As.shape)
        return steer, accel

    def record(self, state, action, target):
        delta = target - self.value(state, action)
        #self.episode_history.append((state + action, delta))
        
        # Hacky, but QLookup needs to do this at every step, not end of episode
        #self.update()
        self.Q[state + action] += self.alpha * delta

        # Stats
        rt = 0.01
        self.error_running_avg = rt * delta **2 + (1 - rt) * self.error_running_avg

    def update(self):
        #for sa, delta in self.episode_history:
        #    self.Q[sa] += self.alpha * delta
            
        #self.episode_history = []

        pass
    
    def plottable(self, axes, pick_max=False):
        ''' Returns a 2D plottable representation for debugging purposes.
            axes a tuple listing which dimensions of the six are to be flattened
            pick_max Whether to show the max or the sum along the flattened 
                dimensions
        '''
        X = self.Q
        X = np.max(X, axis=axes) if pick_max else np.sum(X, axis=axes)
        return X.reshape(X.shape[0], -1).T

    def save_model(self, pref=""):
        util.dump(self.Q, pref+"Q")

    def load_model(self, load_subdir, pref=""):
        self.logger.debug("Loading Q from: %s", load_subdir)
        self.Q = util.load(pref+"Q", load_subdir)
