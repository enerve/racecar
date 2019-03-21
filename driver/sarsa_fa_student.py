'''
Created on Nov 3, 2018

@author: enerve
'''

import logging
import numpy as np

from . import Driver
import util

class SarsaFAStudent(Driver):
    '''
    An agent that learns to drive a car along a track, by observing historical
    episodes and using Sarsa algorithm.
    '''

    def __init__(self,
                 config,
                 gamma,
                 recorder,
                 fa):
        '''
        Constructor
        fa is the value function approximator that learns from the episodes
        '''
        super().__init__(config,
                         fa,
                         mimic_fa)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        self.gamma = gamma  # weight given to predicted future

        self.recorder = recorder # records the steps and feeds the FA

        # TODO: move N to FA?
        # N is the count of visits to state
        self.N = np.zeros((self.num_junctures,
                           self.num_lanes,
                           self.num_speeds,
                           self.num_directions), dtype=np.int32)
                           
        # C is the count of visits to state/action
        self.C = np.zeros((self.num_junctures,
                           self.num_lanes,
                           self.num_speeds,
                           self.num_directions,
                           self.num_steer_positions,
                           self.num_accel_positions), dtype=np.int32)
        # Rs is the average reward at juncture (for statistics)
        self.Rs = np.zeros((self.num_junctures), dtype=np.float)
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

    def prefix(self):
        return "sarsa_" + self.recorder.prefix() + self.fa.prefix()

    def observe_episode(self, steps_history):
        ''' Learns from given episode data. 
            steps_history: list of steps, each a tuples (of S, A, R),
                in chronological order
        '''

        S, A, R = steps_history[0]
        i = 0
        while S is not None:
            i += 1
            S_, A_, R_ = steps_history[i]
        
            I = S + A

            # train fa
            Q_at_next = 0
            if S_ is not None:
                # Find value for the next action / state taken
                Q_at_next = self.fa.value(S_, A_)

            target = R + self.gamma * Q_at_next
            
            self.recorder.step(S, A, target)
            
            # stats
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            delta = (target - self.fa.value(S, A))
            self.avg_delta += 0.02 * (delta - self.avg_delta)

            S, A, R = S_, A_, R_
            
        self.recorder.finish()
                
    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)
        
        if util.checkpoint_reached(ep, num_episodes // 100):
            self.stat_e_100.append(ep)

            self.stat_dlm.append(self.avg_delta)

    def update_fa(self):
        self.fa.update()
    
    def report_stats(self, pref):
        super().report_stats(pref)
        
        util.plot([self.stat_dlm], self.stat_e_100,
                  ["Avg ΔQ student"], pref=pref+"delta",
                  ylim=None)
        #self.fa.report_stats(pref)