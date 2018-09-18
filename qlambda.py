'''
Created on Sep 18, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from driver import Driver
from environment import Environment

class QLambdaDriver(Driver):
    ''' A Driver type that learns using the Watkin's Q-lambda algorithm.
    '''

    def __init__(self, lam,
                 alpha, gamma, explorate,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions,
                 load_filename=None):
        super().__init__(alpha, gamma, explorate,
                         num_junctures,
                         num_lanes,
                         num_speeds,
                         num_directions,
                         num_steer_positions,
                         num_accel_positions,
                         load_filename)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.lam = lam

    def pick_action(self, S, run_best):
        if run_best:
            epsilon = 0
        else:
            n = self.N[S]
            N0 = self.explorate
            epsilon = N0 / (N0 + n)
      
        if random.random() >= epsilon:
            # Pick best
            As = self.Q[S]            
            steer, accel = np.unravel_index(np.argmax(As, axis=None), As.shape)
            
            #debugging
            #if run_best:
            #    my_s, my_a = MY_IDEAL_A[S[0]]
            #    self.logger.debug("I prefer: %d whose Q is %0.2f or %0.2f ",
            #                      my_s, As[my_s, 1], As[my_s, 2])
            #    self.logger.debug("  Chosen: %d, %d whose Q is %0.2f ... at %s", 
            #                      steer, accel, As[steer, accel], S)
        else:
            r = random.randrange(
                self.num_steer_positions * self.num_accel_positions)
            steer = r % self.num_steer_positions
            accel = r // self.num_steer_positions
        
        return (steer, accel)
    
    def run_episode(self, track, car, run_best=False):
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=run_best)
        S = environment.state_encoding() # 0 2 0 0
        total_R = 0
        
        while S is not None:
            A = self.pick_action(S, run_best)
            
            I = self.q_index(S, A)

            R, S_ = environment.step(A)

            target = R + self.gamma * self.max_at_state(S_)
            self.Q[I] += self.alpha * (target - self.Q[I])
            self.C[I] += 1
            self.N[S] += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            
            S = S_
            total_R += R
            
        return total_R, environment
