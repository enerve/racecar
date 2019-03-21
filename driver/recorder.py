'''
Created on 19 Mar 2019

@author: enerve
'''

class Recorder():

    def step(self, state, action, target):
        ''' Record the given step with the inputs (state, action) and the target
            value to be learned for those inputs.
        '''
        pass

    def finish(self):
        ''' Reset after finishing an episode
        '''
        pass
        