'''
Created on 19 Mar 2019

@author: enerve
'''

from driver import Recorder

class LambdaFERecorder(Recorder):
    def __init__(self, fa, feature_eng, lam):
        self.fa = fa
        self.feature_eng = feature_eng
        self.lam = lam
        self.E = None

    def step(self, state, action, target):
        ''' Record the given step with the inputs (state, action) and the target
            value to be learned for those inputs.
        '''
        x = self.feature_eng.x_adjust(*state, *action)
        if self.E is None:
            self.E = x
        else:
            self.E *= self.lam
            self.E += x #the derivative
        self.fa.add_data(S, A, target)

    def finish(self):
        ''' Reset after finishing an episode
        '''
        self.E = None
