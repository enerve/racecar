'''
Created on 30 Apr 2019

@author: enerve
'''

class ExplorationStrategy(object):
    '''
    classdocs
    '''

    def __init__(self, config, explorate, fa):
        '''
        Constructor
        '''
        # TODO: explorate should not be in this base class
        self.explorate = explorate # Inclination to explore, e.g. 0, 10, 1000
        self.fa = fa

        
    def pick_action(self, S, moves):
        pass
    
    def store_exploration_state(self, pref=""):
        pass
    
    def load_exploration_state(self, subdir, pref=""):
        pass
    
    def collect_stats(self, ep):
        pass
    
    def report_stats(self, ep):
        pass