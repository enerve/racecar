'''
Created on May 21, 2019

@author: enerve
'''

import logging

from .explorer import Explorer

class FAExplorer(Explorer):
    '''
    A player that plays using the given exploration strategy, collecting
    experience data in the process.
    '''

    def __init__(self,
                 config,
                 exploration_strategy):
        
        super().__init__(config)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.exploration_strategy = exploration_strategy

    def prefix(self):
        es_prefix = self.exploration_strategy.prefix()
        return "e%s" % es_prefix

    # ---------------- Single episode ---------------

    def _choose_action(self):
        ''' Agent to choose the next action '''
        # Choose action on-policy
        A = self.exploration_strategy.pick_action(self.S, self.moves)
        return A
        
    # ----------- Stats -----------

    def collect_stats(self, episode, ep, num_episodes):
        super().collect_stats(ep, num_episodes)

        self.exploration_strategy.collect_stats(ep)

    def report_stats(self, pref):
        super().report_stats(pref)

        self.exploration_strategy.report_stats(pref)
