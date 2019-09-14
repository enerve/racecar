'''
Created on Apr 30, 2019

@author: enerve
'''

import logging
import numpy as np

from .agent import Agent
from coindrop.lookahead_ab_agent import LookaheadABAgent

class FAAgent(Agent):
    '''
    An agent that chooses the best action using an existing Function Approximator
    '''

    def __init__(self,
                 config,
                 fa):
        self.fa = fa

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # to compare self with simple LA agents
        #TODO: self.buddy = LookaheadABAgent(config, 2)
        
    def prefix(self):
        pref = "faP_" + self.fa.prefix()
        return pref

    # ---------------- Single episode ---------------

    def init_episode(self, initial_state):#, initial_heights):
        super().init_episode(initial_state)#, initial_heights)
        self.S = np.copy(initial_state)

        #TODO: self.buddy.init_episode(initial_state, initial_heights)
        
    def see_outcome(self, reward, new_state, 
                    #new_heights, 
                    moves=0):
        super().see_outcome(reward, new_state, #new_heights, 
                            moves)
        self.S = new_state

        #TODO: self.buddy.see_move(reward, new_state, new_heights, moves)

    def next_action(self):
        ''' Agent's turn. Chooses the next action '''
                
        # Choose action on-policy
        A, val, vals = self.fa.best_action(self.S)

        # For debugging purposes:
        # Check opponent's value for the current state/action
#         if self.moves > 0:
#             opp_val = self.fa.bound_value(-self.S)
#             adiff = abs(opp_val + val)
#             if adiff > 0.5:
#                 self.logger.debug("Opphunch: %0.2f vs %0.2f (diff=%0.2f)", opp_val, val,
#                                   abs(opp_val + val))
# #                 self.logger.debug("%s", self.S)
        
        #TODO: self.debug_buddy_A = self.buddy.next_move()
        self.debug_agent_A = A
        self.debug_prev_S = self.S#.copy()
        self.debug_agent_vals = vals

        return A
    
    def episode_over(self):
        ''' Wrap up episode  '''

        # Record results of episode end
        #TODO: self.buddy.episode_over()
        
#         if self.G <= 0 and self.debug_buddy_A != self.debug_agent_A:
#             self.logger.debug("Different action chosen by buddy: %d rather than %d",
#                               self.debug_buddy_A, self.debug_agent_A)
#             self.logger.debug("  for state: \n%s", self.debug_prev_S)
#             self.logger.debug("  vals: \n%s", [round(x, 2) for x in self.debug_agent_vals])
        pass
