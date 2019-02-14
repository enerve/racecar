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
    An agent that drives a car along a track
    '''

    def __init__(self,
                 fa,
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
        
        self.fa = fa

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


    def restart_exploration(self):
        pass
    
    def run_best_episode(self, track, car, use_test_fa=False):
        ''' Runs an episode picking the best actions based on the driver's
            function approximator
        '''
        fa = self.fa if not use_test_fa else self.fa_test
        
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=True)
        total_R = 0
        steps_history = []
        
        S = environment.state_encoding()
        A = fa.best_action(S)
        A_ = None
        while S is not None:
            R, S_ = environment.step(A)

            steps_history.append((S, A, R))
            
            if S_ is not None:
                A_ = fa.best_action(S_)

            S, A = S_, A_
            total_R += R
            
        steps_history.append((None, None, None))
            
        return total_R, environment, steps_history
    
    def run_episode(self, track, car):
        pass
    
    def collect_stats(self, ep, num_episodes):
        pass
    
    def save_stats(self, pref=None):
        pass

    def load_stats(self, subdir, pref=None):
        pass

    def report_stats(self, pref):
        pass

    def save_model(self, pref=""):
        self.fa.save_model(pref)

    def load_model(self, load_subdir, pref=""):
        self.fa.load_model(load_subdir, pref)
