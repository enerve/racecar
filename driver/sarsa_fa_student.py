'''
Created on Nov 3, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from . import Driver
from environment import Environment
import util

class SarsaFAStudent(Driver):
    '''
    An agent that learns to drive a car along a track, by observing historical
    episodes.
    '''

    def __init__(self, gamma,
                 fa,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions):
        '''
        Constructor
        fa is the value function approximator that gides the episodes and
            leans on the job, using the Sarsa algorithm
        '''
        super().__init__(num_junctures,
                         num_lanes,
                         num_speeds,
                         num_directions,
                         num_steer_positions,
                         num_accel_positions)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        # TODO: pre_alg.. alpha etc
        util.pre_alg = "sarsa_fa_student_%0.1f" % (gamma)
        self.logger.debug("Algorithm: %s", util.pre_alg)

        self.gamma = gamma  # weight given to predicted future

        self.fa = fa
        
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


    def observe_history(self, steps_history):
        ''' Creates training data based on given history. 
            steps_history: tuples (of S, A, R, S_) which are not guaranteed to
                be in sequential order
        '''
        # debugging
        #self.actions_matched = 0
        #self.total_max_actions_picked = 0

        for S, A, R, S_ in steps_history:
            I = S + A

            Q_at_next = 0
            if S_ is not None:
                # For the action "fa" would've taken, find Q at next state
                A_ = self.fa.best_action(S_)
                Q_at_next = self.fa.value(S_, A_)
                #debugging
    #             steer2, accel2 = self.fa_test.best_action(S)
    #             if steer == steer2:
    #                 self.actions_matched += 0.5
    #             if accel == accel2:
    #                 self.actions_matched += 0.5
    #             self.total_max_actions_picked += 1 

            target = R + self.gamma * Q_at_next
            
            self.fa.record(S, A, target)
            
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            if True:# self.C[I] > 1:
                delta = (target - self.fa.value(S, A))
                self.avg_delta += 0.02 * (delta - self.avg_delta)
                
    def update_fa(self):
        self.fa.update()
    
    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)
        
        if util.checkpoint_reached(ep, num_episodes // 100):
            self.stat_e_100.append(ep)

            self.stat_dlm.append(self.avg_delta)
            #self.stat_dlm2.append(self.avg_delta2)
            
            #self.logger.debug("Portion of matched actions: %0.4f",
            #                  self.actions_matched / self.total_max_actions_picked)


    def report_stats(self, pref):
        super().report_stats(pref)
        
        #self.fa_test.report_stats(pref)

        util.plot([self.stat_dlm, self.stat_dlm2], self.stat_e_100,
                  ["Avg ΔQ table", "Avg ΔQ LinReg"], pref="delta",
                  ylim=None)#(-100, 1000))
