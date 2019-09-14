'''
Created on May 15, 2019

@author: enerve
'''

import logging
from really import util

import numpy as np

from .agent import Agent

class Explorer(Agent):
    '''
    Base class for a player that records its experience for future use.
    '''

    def __init__(self,
                 config):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        self.episodes_history = []
        self.test_episodes_history = []
        
        # stats
        self.stats_G = []
        self.sum_G = 0
        self.sum_played = 0

    # ---------------- Single episode ---------------
    
    def prefix(self):
        pass

    def init_episode(self, initial_state):
        super().init_episode(initial_state)
        
        self.S = np.copy(initial_state)
        self.R = 0
        self.steps_history = []
        
    def see_outcome(self, reward, new_state, moves=None):
        ''' Observe the effects on this player of an action taken (possibly by
            another player.)
            reward Reward earned by this player for its last move
        '''
        super().see_outcome(reward, new_state, moves)
        self.R += reward
        self.S = np.copy(new_state)

    def next_action(self):
        ''' Explorer to choose the next action '''
        A = self._choose_action() # Implemented by subclasses

        # Record results R, S of my previous action, and the current action A.
        # (The R of the very first action is irrelevant and is later ignored.)
        self.steps_history.append((self.R, self.S, A))
        self.R = 0  # prepare to collect rewards
        
        return A

    def episode_over(self):
        ''' Wrap up episode '''

        # Record results of episode end
        self.steps_history.append((self.R, None, None))
        
    def save_episode_for_training(self):
        self.episodes_history.append(self.steps_history)

    def save_episode_for_testing(self):
        self.test_episodes_history.append(self.steps_history)

    # ----------- Stats -----------

    def collect_stats(self, ep, total_episodes):
        if (ep+1)% 100 == 0:
            #self.logger.debug("  avg G: %d (%d)" % (self.sum_G / self.sum_played, self.sum_played))
            self.stats_G.append(self.sum_G / self.sum_played)
            self.sum_G = 0
            self.sum_played = 0
            self.live_stats()
        
        self.sum_G += self.G
        self.sum_played += 1 
        
        

    def save_stats(self, pref=""):
        util.dump(np.asarray(self.stats_G, dtype=np.float), "statsG", pref)

    def load_stats(self, subdir, pref=""):
        self.stats_G = util.load("statsG", subdir, pref).tolist()

    def report_stats(self, pref):
        util.plot([self.stats_G],
                  range(len(self.stats_G)),
                  title="recent returns",
                  pref=pref+"_rG")
        
    def live_stats(self):
        #util.plot([self.stats_G],
        #          range(len(self.stats_G)), live=True)
        pass

    def get_episodes_history(self):
        return self.episodes_history

    def get_test_episodes_history(self):
        return self.test_episodes_history
    
    def get_last_episode_history(self):
        return self.steps_history

    def decimate_history(self, dec=1):
        self.episodes_history = [self.episodes_history[i] for i in
                                 range(len(self.episodes_history)) 
                                 if i % 10 >= dec]
        self.test_episodes_history = [self.test_episodes_history[i] for i in 
                                      range(len(self.test_episodes_history)) 
                                      if i % 10 >= dec]

    def store_episode_history(self, fname):
        EH = np.asarray(self.episodes_history)
        util.dump(EH, fname, "EH")
        VEH = np.asarray(self.test_episodes_history)
        util.dump(VEH, fname, "VEH")
 
    def load_episode_history(self, fname, subdir):
        self.logger.debug("Loading episode history from %s %s", subdir, fname)
        self.episodes_history = [[(s[0], s[1], s[2]) for s in sh]
                                 for sh in util.load(fname, subdir, suffix="EH")]
        self.test_episodes_history = [[(s[0], s[1], s[2]) for s in sh] 
                                      for sh in util.load(fname, subdir, suffix="VEH")]

