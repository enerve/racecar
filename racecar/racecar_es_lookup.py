'''
Created on 30 Apr 2019

@author: enerve
'''

import logging
import numpy as np
import random
from really.agent import ExplorationStrategy

class RacecarESLookup(ExplorationStrategy):
    '''
    Exploration strategy that uses a lookup table to store frequency of visits
    '''

    def __init__(self, config, explorate, fa):
        '''
        Constructor
        '''
        super().__init__(config, explorate, fa)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.N = np.zeros((config.NUM_JUNCTURES,
                           config.NUM_LANES,
                           config.NUM_SPEEDS,
                           config.NUM_DIRECTIONS), dtype=np.int32)

        self.C = np.zeros((config.NUM_JUNCTURES,
                           config.NUM_LANES,
                           config.NUM_SPEEDS,
                           config.NUM_DIRECTIONS,
                           config.NUM_STEER_POSITIONS,
                           config.NUM_ACCEL_POSITIONS), dtype=np.int32)
        
        self.stats_a_count = np.zeros((3, 3))
        self.stats_ba_count = np.zeros((3, 3))

    def prefix(self):
        return "lookup"

    def pick_action(self, S, num_steps):
        n = self.N[tuple(S)]
        N0 = self.explorate
        epsilon = N0 / (N0 + n)

        if random.random() >= epsilon:
            # Pick best
            action = self.fa.best_action(S)[0]
            self.stats_ba_count[action] += 1
        else:
            action = self.fa.random_action(S)

        self.stats_a_count[action] += 1
        
        self.N[tuple(S)] += 1

        self.C[tuple(S) + action] += 1

        return action

    def collect_stats(self, ep):
        if ep % 1000 == 0:
            sum_ab = np.sum(self.stats_ba_count)
            self.logger.debug("Best Action count:\n%s", self.stats_ba_count/sum_ab)

    def report_stats(self, pref=''):
        sum_a = np.sum(self.stats_a_count)
        self.logger.debug("Action count:\n%s", self.stats_a_count/sum_a)
        sum_ab = np.sum(self.stats_ba_count)
        self.logger.debug("Best Action count:\n%s", self.stats_ba_count/sum_ab)
