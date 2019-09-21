'''
Created on 14 Feb 2019

@author: enerve
'''

from really import util
from really.function import FeatureEng

import math
import torch
import random

class RacecarSAFeatureEng(FeatureEng):
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


