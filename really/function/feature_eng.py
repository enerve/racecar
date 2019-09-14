'''
Created on 14 Feb 2019

@author: erwin
'''

class FeatureEng(object):
    '''
    Base class for engineered features input vectors
    '''

    def num_actions(self):
        pass

    def prefix(self):
        pass

    def x_adjust(self, S):
        pass

    def value_from_output(self, net_output):
        pass
    
    def output_for_value(self, value):
        pass

    def a_index(self, a_tuple):
        pass

    def action_from_index(self, a_index):
        pass

    def valid_actions_mask(self, B):
        pass

    def random_action(self, state):
        pass
    
    def prepare_data_for(self, S, a, target):
        pass