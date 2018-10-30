'''
Created on Oct 24, 2018

@author: enerve
'''

import logging
import numpy as np

from q_driver import QDriver
from environment import Environment
import util

class QLambdaDriver(QDriver):
    '''
    An agent that learns to drive a car along a track, optimizing using 
    Q-lambda
    '''

    def __init__(self, lam, alpha, gamma, explorate,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions):

        '''
        Constructor
        '''
        super().__init__(alpha, gamma, explorate,
                         num_junctures,
                         num_lanes,
                         num_speeds,
                         num_directions,
                         num_steer_positions,
                         num_accel_positions)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_alg = "q_lambda_%d_%0.1f_%0.1f_%0.1f" % (explorate, alpha,
                                                          gamma, lam)
        
        self.lam = lam
        
        self.num_resets = 0
        self.num_steps = 0

        self.eligible_mult = [(lam * gamma) ** i for i in range(num_junctures)]
        self.eligible_states = [None for i in range(num_junctures)]

    def run_episode(self, track, car, run_best=False):
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=run_best)
        S = environment.state_encoding()
        total_R = 0
        
        # Eligibility
        num_E = 0
        self.num_resets += 1
        Q_at_chosen_next = 0  # initialization doesn't matter
        
        
        A = self._pick_action(S, run_best)
        A_ = None
        while S is not None:
            
            I = S + A

            R, S_ = environment.step(A)

            Q_at_max_next = self.max_at_state(S_)
            if S_ is not None:
                A_ = self._pick_action(S_, run_best)
                Q_at_chosen_next = self.Q[S_ + A_]

            target = R + self.gamma * Q_at_max_next
            delta = (target - self.Q[I])
            self.eligible_states[num_E] = I
            num_E += 1
            for i in range(num_E):
                e_I = self.eligible_states[i]
                self.Q[e_I] += self.alpha * delta * self.eligible_mult[num_E-i-1]
            self.C[I] += 1
            self.N[S] += 1

            if Q_at_chosen_next != Q_at_max_next:
                # The policy diverted from Q* policy so zero out eligibilities
                num_E = 0
                self.num_resets += 1

            # Collect stats
            self.num_steps += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            if self.C[I] > 1:
                self.avg_delta += 0.02 * (delta - self.avg_delta)
            
            S, A = S_, A_
            total_R += R
            
        return total_R, environment

    def run_episode_slow(self, track, car, run_best=False):
        ''' Uses a slower implementation of Q-lambda's Eligibility
        '''
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=run_best)
        S = environment.state_encoding()
        total_R = 0
         
        # Eligibility
        E = np.zeros_like(self.Q)
        #num_E = 0
        self.num_resets += 1
        Q_at_chosen_next = 0  # initialization doesn't matter
         
         
        A = self.pick_action(S, run_best)
        while S is not None:
             
            I = S + A
 
            R, S_ = environment.step(A)
 
            Q_at_max_next = self.max_at_state(S_)
            if S_ is not None:
                A_ = self.pick_action(S_, run_best)
                Q_at_chosen_next = self.Q[S_ + A_]
 
            target = R + self.gamma * Q_at_max_next
            delta = (target - self.Q[I])
            E[I] += 1
            self.Q += self.alpha * delta * E
            self.C[I] += 1
            self.N[S] += 1
 
            if Q_at_chosen_next == Q_at_max_next:
                E *= self.gamma * self.lam
            else:
                # The policy diverted from Q* policy so zero out eligibilities
                E = np.zeros_like(self.Q)
                #num_E = 0
                self.num_resets += 1
 
            # Collect stats
            self.num_steps += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            if self.C[I] > 1:
                self.avg_delta += 0.02 * (delta - self.avg_delta)
             
            S, A = S_, A_
            total_R += R
             
        return total_R, environment

    def save_stats(self, pref=None):
        super().save_stats(pref)
        
        A = np.asarray([
            self.num_steps,
            self.num_resets
            ], dtype=np.float)
        util.dump(A, "dstatsLam", pref)

    def load_stats(self, subdir, pref=None):
        super().load_stats(subdir, pref)
        
        A = util.load("dstatsLam", subdir)
        self.num_steps = A[0]
        self.num_resets = A[1]
    
    def report_stats(self, pref):
        super().report_stats(pref)
        
        self.logger.debug("Average length of eligibility trace: %0.2f", 
                          self.num_steps / self.num_resets)