'''
Created on 14 Feb 2019

@author: enerve
'''

from really import util
from really.function import FeatureEng

import math
import torch
import random

class RacecarFeatureEng(FeatureEng):
    '''
    Feature engineering base class for Racecar
    '''


    def __init__(self,
                 config):
        '''
        Constructor
        '''
        
        # states
        self.num_junctures = config.NUM_JUNCTURES
        self.num_lanes = config.NUM_LANES
        self.num_directions = config.NUM_DIRECTIONS 
        self.num_speeds = config.NUM_SPEEDS
        # actions
        self.num_steer_positions = config.NUM_STEER_POSITIONS
        self.num_accel_positions = config.NUM_ACCEL_POSITIONS
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')


    def num_actions(self):
        return self.num_steer_positions * self.num_accel_positions
        
    def value_from_output(self, net_output):
        return net_output
    
    def output_for_value(self, value):
        return value

    def a_index(self, a_tuple):
        steer, accel = a_tuple
        return self.num_accel_positions * steer + accel

    def action_from_index(self, a_index):
        return (a_index // self.num_accel_positions,
                a_index % self.num_accel_positions)
        
    def valid_actions_mask(self, B):
        valid_actions_mask = torch.ones((self.num_actions())).to(self.device).float()
        return valid_actions_mask

    def random_action(self, state):
        r = random.randrange(
            self.num_steer_positions * self.num_accel_positions)
        return self.action_from_index(r)
    
    def prepare_data_for(self, S, a, target):
        hist_x, hist_t, hist_mask = [], [], []
        
        t = self.output_for_value(target)
        x = self.x_adjust(S)
        ai = self.a_index(a)
        m = self.teye[ai].clone()
    
        hist_x.append(x)
        hist_t.append(t)
        hist_mask.append(m)
                
        return hist_x, hist_t, hist_mask
