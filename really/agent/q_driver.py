'''
Created on Sep 6, 2018

@author: enerve
'''

import logging
import numpy as np

from . import SADriver
from environment import Environment
import util

class QDriver(SADriver):
    '''
    An agent that learns to drive a car along a track, optimizing using 
    Q-learning
    (Deprecated in favor of QFADriver.)
    Uses a Q-table-lookup function approximator.
    '''

    def __init__(self, alpha, gamma, explorate,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions):
        '''
        Constructor
        '''
        super().__init__(gamma,
                         explorate,
                         num_junctures,
                         num_lanes,
                         num_speeds,
                         num_directions,
                         num_steer_positions,
                         num_accel_positions)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_alg = "qlearn_%d_%0.1f_%0.1f" % (explorate, alpha, gamma)
        self.logger.debug("Algorithm: %s", util.pre_alg)

        self.alpha = alpha  # learning rate for updating Q

    def run_episode(self, track, car, run_best=False):
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=run_best)
        S = environment.state_encoding()
        total_R = 0
        
        while S is not None:
            A = self._pick_action(S, run_best)
            
            I = S + A

            R, S_ = environment.step(A)

            target = R + self.gamma * self._max_at_state(S_)
            delta = (target - self.Q[I])
            self.Q[I] += self.alpha * delta
            self.C[I] += 1
            self.N[S] += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            if self.C[I] > 1:
                self.avg_delta += 0.02 * (delta - self.avg_delta)

            S = S_
            total_R += R
            
        return total_R, environment
