'''
Created on Nov 3, 2018

@author: enerve
'''

class ValueFunction(object):
    '''
    Base class for action-value function approximators
    '''

    def value(self, state, action):
        pass

    def best_action(self, S):
        pass

    def record(self, state, action, delta):
        pass

    def update(self):
        pass
    
    def collect_stats(self, ep):
        pass
    
    def report_stats(self, pref):
        pass