'''
Created on Sep 6, 2018

@author: enerve
'''

import logging
import numpy as np
import random
from driver import Driver

from environment import Environment
import util

class ManualDriver(Driver):
    '''
    An agent that drives with the given fixed sequence of actions.
    '''

    def __init__(self,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions,
                 action_sequence):
        '''
        Constructor
        '''
        super().__init__(num_junctures,
                         num_lanes,
                         num_speeds,
                         num_directions,
                         num_steer_positions,
                         num_accel_positions)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_alg = "test_driver"
        
        self.action_sequence = action_sequence
        
    def run_episode(self, track, car):
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=True)

        total_R = 0
        for A in self.action_sequence:
            R, S_ = environment.step(A)
            self.logger.debug(" A:%s  R:%d    S:%s" % (A, R, S_,))
            if S_ is None:
                break
            total_R += R

        return total_R, environment
    
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
