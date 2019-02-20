'''
Created on Jan 7, 2019

@author: enerve
'''

import logging
import numpy as np

from . import Driver
import util

class QLambdaFAStudent(Driver):
    '''
    An agent that learns to drive a car along a track by observing episodes,
    and optimizing using Q(λ)
    '''

    def __init__(self, lam, gamma,
                 fa,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions,
                 mimic_fa):
        '''
        Constructor
        fa is the value function approximator that learns from the episodes
        '''
        super().__init__(fa,
                         num_junctures,
                         num_lanes,
                         num_speeds,
                         num_directions,
                         num_steer_positions,
                         num_accel_positions,
                         mimic_fa)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        self.lam = lam      # lookahead parameter
        self.gamma = gamma  # weight given to predicted future
        
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

    def prefix(self):
        return "q_lambda_l%0.2f_" % self.lam + self.fa.prefix()

    def observe_episode(self, steps_history):
        ''' Collects training data based on given episode data. 
            steps_history: list of steps, each a tuples (of S, A, R),
                in chronological order
        '''

        # Eligibility
        num_E = 0
        Q_at_chosen_next = 0  # initialization doesn't matter
        
        S, A, R = steps_history[0]
        hi = 0
        while S is not None:
            hi += 1
            S_, A_, R_ = steps_history[hi]
            
            I = S + A
            
            Q_at_max_next = 0
            if S_ is not None:
                # off-policy
                max_A = self.fa.best_action(S_)
                Q_at_max_next = self.fa.value(S_, max_A)
                # on-policy
                Q_at_chosen_next = self.fa.value(S_, A_)

            target = R + self.gamma * Q_at_max_next
            
            curr_value = self.fa.value(S, A)  #TODO: =previous Q_at_chosen_next
            self.eligible_states[num_E] = (S, A)
            self.eligible_state_target[num_E] = curr_value

            delta = (target - curr_value)
            num_E += 1
            for i in range(num_E):
                self.eligible_state_target[i] += delta * self.eligible_mult[num_E-i-1]
            self.C[I] += 1
            self.N[S] += 1
            
            if Q_at_chosen_next != Q_at_max_next:
                # The policy diverted from Q* policy so restart eligibilities
                # But first, flush the eligibility updates into the FA
                self._record_eligibles(num_E)
                num_E = 0
            
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            delta = (target - self.fa.value(S, A))
            self.avg_delta += 0.02 * (delta - self.avg_delta)
            
            S, A, R = S_, A_, R_

        self._record_eligibles(num_E)
    
    def _record_eligibles(self, num_E):
        for i in range(num_E):
            S, A = self.eligible_states[i]
            target = self.eligible_state_target[i]
            self.fa.record(S, A, target)
    
    def update_fa(self):
        self.fa.update()

    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)
        
        if util.checkpoint_reached(ep, num_episodes // 100):
            self.stat_e_100.append(ep)

            self.stat_dlm.append(self.avg_delta)

    def report_stats(self, pref):
        super().report_stats(pref)
        util.plot([self.stat_dlm], self.stat_e_100,
                  ["Avg ΔQ student fa"], pref=pref+"delta",
                  ylim=None)
        self.fa.report_stats(pref)
