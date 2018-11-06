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

    def update(self, state, action, alpha, delta):
        pass
