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
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')


