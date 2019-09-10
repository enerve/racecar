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
                 config,
                 fa,
                 mimic_fa):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_alg = "driverbase"
        
        self.fa = fa

        self.num_junctures = config.NUM_JUNCTURES
        self.num_lanes = config.NUM_LANES
        self.num_directions = config.NUM_DIRECTIONS 
        self.num_speeds = config.NUM_SPEEDS
        self.num_steer_positions = config.NUM_STEER_POSITIONS
        self.num_accel_positions = config.NUM_ACCEL_POSITIONS
        self.mimic_fa = mimic_fa

        # C is the count of visits to state/action
        self.C = np.zeros((self.num_junctures,
                           self.num_lanes,
                           self.num_speeds,
                           self.num_directions,
                           self.num_steer_positions,
                           self.num_accel_positions), dtype=np.int32)
        # Rs is the average reward at juncture (for statistics)
        self.Rs = np.zeros((self.num_junctures), dtype=np.float)


    def restart_exploration(self):
        pass
    
    def run_best_episode(self, track, car, use_mimic=False):
        ''' Runs an episode picking the best actions based on the driver's
            function approximator
        '''
        fa = self.fa if not use_mimic else self.mimic_fa
        
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
    
    def live_stats(self):
        pass

    def save_model(self, pref=""):
        self.fa.save_model(pref)
        if self.mimic_fa:
            self.mimic_fa.save_model(pref+"_mimic")

    def load_model(self, load_subdir, pref=""):
        self.fa.load_model(load_subdir, pref)
