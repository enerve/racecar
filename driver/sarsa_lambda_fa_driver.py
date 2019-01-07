'''
Created on Nov 13, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from . import Driver
from environment import Environment
import util

class SarsaLambdaFADriver(Driver):
    '''
    An agent that learns to drive a car along a track, optimizing using Sarsa(λ)
    '''

    TEST_FA = False

    def __init__(self, lam, gamma, explorate,
                 fa,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions):
        '''
        Constructor
        fa is the value function approximator that guides the episodes and
            learns on the job, using the Q algorithm
        '''
        super().__init__(fa,
                         num_junctures,
                         num_lanes,
                         num_speeds,
                         num_directions,
                         num_steer_positions,
                         num_accel_positions)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        # TODO: pre_alg.. alpha etc
        util.pre_alg = "sarsa_lambda_fa_driver_%d_%0.1f" % (explorate, gamma)
        self.logger.debug("Algorithm: %s", util.pre_alg)

        self.lam = lam      # lookahead parameter
        self.gamma = gamma  # weight given to predicted future
        self.explorate = explorate # Inclination to explore, e.g. 0, 10, 1000

        self.fa = fa
        
        if self.TEST_FA:
            # For testing the function-appromiating capabilities of any new
            # potential FAs.
            from function import PolynomialRegression
            self.fa_test = PolynomialRegression(
                            0.01, # alpha ... #4e-5 old alpha without batching
                            0.02, # regularization constant
                            256, # batch_size
                            250, # max_iterations
                            num_junctures,
                            num_lanes,
                            num_speeds,
                            num_directions,
                            num_steer_positions,
                            num_accel_positions)        
            self.avg_delta2 = 0
            self.stat_dlm2 = []

        self.eligible_mult = [(lam * gamma) ** i for i in range(num_junctures)]
        self.eligible_states = [None for i in range(num_junctures)]
        self.eligible_state_target = [0 for i in range(num_junctures)]
        
        # TODO: move N to FA?
        # N is the count of visits to state
        self.N = np.zeros((self.num_junctures,
                           self.num_lanes,
                           self.num_speeds,
                           self.num_directions), dtype=np.int32)
                           
        # C is the count of visits to state/action
        self.C = np.zeros((num_junctures,
                           num_lanes,
                           num_speeds,
                           num_directions,
                           num_steer_positions,
                           num_accel_positions), dtype=np.int32)
        # Rs is the average reward at juncture (for statistics)
        self.Rs = np.zeros((num_junctures), dtype=np.float)
        self.avg_delta = 0
        self.restarted = False


        # Stats
        
        self.stat_e_100 = []
        self.stat_qm = []
        self.stat_cm = []
        self.stat_rm = []
        
        self.stat_debug_qv = []
        self.stat_debug_n = []

        # TODO: Move to QLookup
        self.stat_e_200 = []
        self.q_plotter = util.Plotter("Q value at junctures along lanes")
        self.qx_plotter = util.Plotter("Max Q value at junctures along lanes")
        self.c_plotter = util.Plotter("C value at junctures along lanes")

        # track average change in Q, as iterations progress
        self.stat_dlm = []
        
        self.actions_matched = 0
        self.total_max_actions_picked = 0

        self.num_resets = 0
        self.num_steps = 0

    def restart_exploration(self, scale_explorate=1):
        super().restart_exploration()
        
        # N is the count of visits to state
        self.N = np.zeros((self.num_junctures,
                           self.num_lanes,
                           self.num_speeds,
                           self.num_directions), dtype=np.int32)
        self.explorate *= scale_explorate
        self.logger.debug("Restarting with explorate=%d", self.explorate)
        self.avg_delta = 0
        self.restarted = True

    def _pick_action(self, S):
        n = self.N[S]
        N0 = self.explorate
        epsilon = N0 / (N0 + n)
      
        if random.random() >= epsilon:
            # Pick best
            #self.logger.debug(S)
            steer, accel = self.fa.best_action(S)
            
            #debugging
#             steer2, accel2 = self.fa_test.best_action(S)
#             if steer == steer2:
#                 self.actions_matched += 0.5
#             if accel == accel2:
#                 self.actions_matched += 0.5
#             self.total_max_actions_picked += 1 
            
            #debugging
#             if run_best:
# #                 my_s, my_a = MY_IDEAL_A[S[0]]
# #                 self.logger.debug("I prefer: %d whose Q is %0.2f or %0.2f ",
# #                                   my_s, As[my_s, 1], As[my_s, 2])
#                 self.logger.debug("  Chosen: %d, %d whose Q is %0.2f ... at %s", 
#                                   steer, accel, As[steer, accel], S)
#                 self.logger.debug("  Options: %s", As)
        else:
            r = random.randrange(
                self.num_steer_positions * self.num_accel_positions)
            steer, accel = divmod(r, self.num_accel_positions)
        
        return (steer, accel)
    

    def run_episode(self, track, car):
        environment = Environment(track,
                                  car,
                                  self.num_junctures)
        total_R = 0
        steps_history = []
        
        # Eligibility
        num_E = 0
        
        S = environment.state_encoding()
        A = self._pick_action(S)
        A_ = None
        curr_value = self.fa.value(S, A)
        while S is not None:
            R, S_ = environment.step(A)
            
            steps_history.append((S, A, R))
            I = S + A
            
            Q_at_next = 0
            if S_ is not None:
                A_ = self._pick_action(S_)
                Q_at_next = self.fa.value(S_, A_)

            target = R + self.gamma * Q_at_next
            
            # Note: curr_value == self.fa.value(S, A)
            self.eligible_states[num_E] = (S, A)
            self.eligible_state_target[num_E] = curr_value

            delta = (target - curr_value)
            num_E += 1
            for i in range(num_E):
                self.eligible_state_target[i] += delta * self.eligible_mult[
                    num_E-i-1]
            self.C[I] += 1
            self.N[S] += 1
            
            self.num_steps += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            delta = (target - self.fa.value(S, A))
            self.avg_delta += 0.02 * (delta - self.avg_delta)
            if self.TEST_FA:
                delta2 = (target - self.fa_test.value(S, A))
                self.avg_delta2 += 0.02 * (delta2 - self.avg_delta2)
            
            S, A, curr_value = S_, A_, Q_at_next
            total_R += R

        self._record_eligibles(num_E)
        
        steps_history.append((None, None, None))

        return total_R, environment, steps_history
    
    def _record_eligibles(self, num_E):
        for i in range(num_E):
            S, A = self.eligible_states[i]
            target = self.eligible_state_target[i]
            self.fa.record(S, A, target)
            if self.TEST_FA:
                self.fa_test.record(S, A, target)
        self.num_resets += 1
    
    def update_fa(self):
        self.fa.update()
        if self.TEST_FA:
            self.fa_test.update()

        # debugging
        self.actions_matched = 0
        self.total_max_actions_picked = 0

    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)
        
        if util.checkpoint_reached(ep, num_episodes // 100):
            self.stat_e_100.append(ep)

            self.stat_dlm.append(self.avg_delta)
            if self.TEST_FA:
                self.stat_dlm2.append(self.avg_delta2)
            
            #self.logger.debug("Portion of matched actions: %0.4f",
            #                  self.actions_matched / self.total_max_actions_picked)

    def report_stats(self, pref):
        super().report_stats(pref)
        util.plot([self.stat_dlm], self.stat_e_100,
                  ["Avg ΔQ fa"], pref=pref+"delta",
                  ylim=None)

        if self.TEST_FA:
            self.fa_test.report_stats(pref)
            util.plot([self.stat_dlm, self.stat_dlm2], self.stat_e_100,
                      ["Avg ΔQ fa", "Avg ΔQ fa_test"], pref=pref+"delta",
                      ylim=None)

        self.logger.debug("Average length of eligibility trace: %0.2f", 
                          self.num_steps / self.num_resets)