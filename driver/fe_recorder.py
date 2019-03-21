'''
Created on 19 Mar 2019

@author: enerve
'''

from driver import Recorder

class FERecorder(Recorder):
    def __init__(self, fa, feature_eng):
        self.fa = fa
        self.feature_eng = feature_eng

    def step(self, state, action, target):
        ''' Record the given step with the inputs (state, action) and the target
            value to be learned for those inputs.
        '''
        x = self.feature_eng.x_adjust(*state, *action)
        self.fa.add_data(x, target)

