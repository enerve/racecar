'''
Created on May 21, 2019

@author: enerve
'''

import logging
import numpy as np
from really import util

from really.agent import FAExplorer

class RacecarExplorer(FAExplorer):
    '''
    A racecar driver that explores using the given exploration strategy,
    collecting experience data in the process.
    '''

    def __init__(self,
                 config,
                 exploration_strategy):
        
        super().__init__(config, exploration_strategy)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.best_G = -10000

        self.num_junctures = config.NUM_JUNCTURES 
        
        self.stat_ep_m = []
        self.stat_m = []
        self.count_m = None

    # ----------- Stats -----------

    def collect_stats(self, episode, ep, total_episodes):
        super().collect_stats(episode, ep, total_episodes)
        
        if episode.curr_juncture >= 6:
            if self.G > self.best_G:
                self.logger.debug("Ep %d  Juncture %d reached with G=%d T=%d",
                                  ep, episode.curr_juncture, self.G,
                                  episode.total_time_taken())
                self.best_G = self.G

        smooth = total_episodes // 100
        if self.count_m is None:
            self.count_m = np.zeros((smooth, self.num_junctures + 1), dtype=np.int32)
            self.Eye = np.eye(self.num_junctures + 1)

        self.count_m[ep % smooth] = self.Eye[episode.curr_juncture]
        if util.checkpoint_reached(ep, total_episodes // 200):
            self.stat_ep_m.append(ep)
            self.stat_m.append(self.count_m.sum(axis=0))

    def report_stats(self, pref):
        super().report_stats(pref)
        
        S_m = np.array(self.stat_m).T
        labels = ["ms %d" % i for i in range(len(S_m))]
        util.plot(S_m, self.stat_ep_m, labels, "Max juncture reached", pref="ms")
