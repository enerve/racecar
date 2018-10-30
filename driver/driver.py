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


    def restart_exploration(self):
        pass
    
    def run_episode(self, track, car, run_best=False):
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
        pass

    def load_model(self, load_subdir):
        pass
