'''
Created on 19 Mar 2019

@author: enerve
'''

from driver import Recorder

class DirectRecorder(Recorder):
    def __init__(self, fa):
        self.fa = fa

    def prefix(self):
        return "rDir_"

    def step(self, state, action, target):
        ''' Record the given step with the inputs (state, action) and the target
            value to be learned for those inputs.
        '''
        self.fa.add_data(state, action, target)

