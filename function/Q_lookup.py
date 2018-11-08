'''
Created on Nov 3, 2018

@author: enerve
'''

import logging
import numpy as np
from . import ValueFunction

class QLookup(ValueFunction):
    '''
    A function approximator that stores value as table-lookups.
    '''

    def __init__(self,
                 alpha,
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

        # Q is the learned value of a state/action
        self.Q = np.zeros((num_junctures,
                           num_lanes,
                           num_speeds,
                           num_directions,
                           num_steer_positions,
                           num_accel_positions))
        
        self.alpha = alpha
        self.episode_history = []
        
    def value(self, state, action):
        return self.Q[state + action]

    def best_action(self, S):
        As = self.Q[S]
        steer, accel = np.unravel_index(np.argmax(As, axis=None), As.shape)
        return steer, accel

    def record(self, state, action, target):
        delta = target - self.value(state, action)
        self.episode_history.append((state + action, delta))
        
        # Hacky, but QLookup needs to do this at every step, not end of episode
        self.update()

    def update(self):
        for sa, delta in self.episode_history:
            self.Q[sa] += self.alpha * delta
            
        self.episode_history = []