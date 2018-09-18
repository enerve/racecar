'''
Created on Sep 6, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from environment import Environment
import util

class Driver(object):
    '''
    An agent that learns to drive a car along a track, optimizing using 
    Q-learning
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
    
        util.pre_alg = "driverbase"
        
        self.num_junctures = num_junctures
        self.num_lanes = num_lanes
        self.num_speeds = num_speeds
        self.num_directions = num_directions
        self.num_steer_positions = num_steer_positions
        self.num_accel_positions = num_accel_positions

        # C is the count of visits to state/action
        self.C = np.zeros((num_junctures,
                           num_lanes,
                           num_speeds,
                           num_directions,
                           num_steer_positions,
                           num_accel_positions), dtype=np.int32)
        # Rs is the average reward at juncture (for statistics)
        self.Rs = np.zeros((num_junctures), dtype=np.float)

        self.stat_e_100 = []
        self.stat_qm = []
        self.stat_cm = []
        self.stat_rm = []
        
        self.stat_debug_qv = []
        self.stat_debug_n = []

        self.stat_e_200 = []
        self.q_plotter = util.Plotter("Q value at junctures along lanes")
        self.qx_plotter = util.Plotter("Max Q value at junctures along lanes")
        self.c_plotter = util.Plotter("C value at junctures along lanes")

        # track max Q per juncture, as iterations progress
        self.stat_juncture_maxQ = []

    def restart_exploration(self):
        pass
    
    def pick_action(self, S, run_best):  
        pass
    
    def run_episode(self, track, car, run_best=False):
        #         environment = Environment(track,
        #                                   car,
        #                                   self.num_junctures,
        #                                   should_record=run_best)
        #         S = environment.state_encoding() # 0 2 0 0
        #         total_R = 0
        #         
        #         while S is not None:
        #             A = self.pick_action(S, run_best)
        #             
        #             I = self.q_index(S, A)
        # 
        #             R, S_ = environment.step(A)
        # 
        #             # Experiment: testing modified alpha
        #             factor = 1
        # #             if self.restarted:
        # #                 n = self.N[S]
        # #                 N0 = self.explorate
        # #                 factor = n / (N0 + n)
        # 
        #             target = R + self.gamma * self.max_at_state(S_)
        #             self.Q[I] += self.alpha * (target - self.Q[I]) * factor
        #             self.C[I] += 1
        #             self.N[S] += 1
        #             if S_ is not None:
        #                 self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
        #             
        #             S = S_
        #             total_R += R
        #             
        #         return total_R, environment
        pass
    
    def collect_stats(self, ep, num_episodes):
        pass
    
    def report_stats(self, pref):
        pass

    def saveToFile(self, pref=""):
        pass

            
