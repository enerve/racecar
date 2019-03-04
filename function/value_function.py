'''
Created on Nov 3, 2018

@author: enerve
'''

class ValueFunction(object):
    '''
    Base class for action-value function approximators
    '''

    def prefix(self):
        pass
    
    def value(self, state, action):
        pass

    def best_action(self, state):
        pass

    def record(self, state, action, target):
        pass

    def update(self):
        pass

    def train(self):
        pass
    
    def test(self):
        pass

    def collect_stats(self, ep):
        pass
    
    def report_stats(self, pref):
        pass
    
    def store_training_data(self, fname):
        pass

    def load_training_data(self, fname, subdir):
        pass

    def save_model(self, pref=""):
        pass

    def load_model(self, load_subdir, pref=""):
        pass