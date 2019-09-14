'''
Created on 12 Sep 2019

@author: enerve
'''

from .exploration_strategy import ExplorationStrategy

class ESBest(ExplorationStrategy):
    '''
    An exploration strategy that always simply picks the best action
    '''

    def __init__(self, config, fa):
        '''
        Constructor
        '''
        super().__init__(config, 0, fa)
        
    def prefix(self):
        return "best"
        
    def pick_action(self, S, moves):
        return self.fa.best_action(S)[0]
