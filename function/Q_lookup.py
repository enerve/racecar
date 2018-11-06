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
        
    def value(self, state, action):
        return self.Q[state + action]

    def best_action(self, S):
        As = self.Q[S]
        steer, accel = np.unravel_index(np.argmax(As, axis=None), As.shape)
        return steer, accel

    def update(self, state, action, alpha, delta):
        self.Q[state + action] += alpha * delta
