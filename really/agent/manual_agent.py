'''
Created on Sep 6, 2018

@author: enerve
'''

import logging
from .agent import Agent

class ManualAgent(Agent):
    '''
    An agent that follows the given fixed sequence of actions.
    '''

    def __init__(self,
                 config,
                 action_sequence):
        '''
        Constructor
        '''
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.action_sequence = action_sequence        

    def init_episode(self, initial_state):#, initial_heights):
        ''' Initialize for a new episode
        '''
        super().init_episode(initial_state)

        self.actions_taken = 0
        self.A = None

    def see_outcome(self, reward, new_state, moves=0):
        super().see_outcome(reward, new_state, moves)
        self.logger.debug(" A:%s  R:%d    S:%s" % (self.A, self.G, new_state))

    def _choose_action(self):
        # Pick the next action in the given sequence
        self.A = self.action_sequence[self.actions_taken]
        self.actions_taken += 1
        
        return self.A
