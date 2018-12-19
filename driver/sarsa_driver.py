'''
Created on Oct 30, 2018

@author: enerve
'''

import logging
import numpy as np

from . import SADriver
from environment import Environment
import util

class SarsaDriver(SADriver):
    '''
    An agent that learns to drive a car along a track, optimizing using 
    Sarsa
    (Deprecated in favor of SarsaFADriver.)
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
    
        util.pre_alg = "sarsa_%d_%0.1f_%0.1f" % (explorate, alpha, gamma)
        self.logger.debug("Algorithm: %s", util.pre_alg)

        self.alpha = alpha  # learning rate for updating Q

    def run_episode(self, track, car, run_best=False):
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=run_best)
        S = environment.state_encoding()
        total_R = 0
        
        A_ = None
        A = self._pick_action(S, run_best)
        while S is not None:
            R, S_ = environment.step(A)
            
            I = S + A

            Q_at_next = 0
            if S_ is not None:
                A_ = self._pick_action(S_, run_best)
                Q_at_next = self.Q[S_ + A_]

            target = R + self.gamma * Q_at_next
            delta = (target - self.Q[I])
            self.Q[I] += self.alpha * delta
            self.C[I] += 1
            self.N[S] += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            if self.C[I] > 1:
                self.avg_delta += 0.02 * (delta - self.avg_delta)

            S, A = S_, A_
            total_R += R
            
        return total_R, environment
