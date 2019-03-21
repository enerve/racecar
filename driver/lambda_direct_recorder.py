'''
Created on 19 Mar 2019

@author: enerve
'''

from driver import Recorder

class LambdaDirectRecorder():
    def __init__(self, gamma, fa, lam):
        self.fa = fa
        self.gamma = gamma
        self.lam = lam

        self.eligible_mult = []
        self.eligible_states = []
        self.eligible_state_target = []
        self.num_E = 0

    def prefix(self):
        return "rLDir_l%0.2f_" % self.lam
    
    def step(self, S, A, target):
        ''' Record the given step with the inputs (state, action) and the target
            value to be learned for those inputs.
        '''        
        curr_value = self.fa.value(S, A)

        #TODO: unnecessarily complicated. maybe skip pre-computing mult.
        if len(self.eligible_mult) == self.num_E:
            self.eligible_mult.append((self.lam * self.gamma) ** self.num_E)
            self.eligible_states.append((S, A))
            self.eligible_state_target.append(curr_value)
        else:
            self.eligible_states[self.num_E] = (S, A)
            self.eligible_state_target[self.num_E] = curr_value

        delta = (target - curr_value)
        self.num_E += 1
        for i in range(self.num_E):
            self.eligible_state_target[i] += delta * self.eligible_mult[
                self.num_E-i-1]

    def finish(self):
        ''' Reset after finishing an episode
        '''
        for i in range(self.num_E):
            S, A = self.eligible_states[i]
            target = self.eligible_state_target[i]
            self.fa.add_data(S, A, target)
        self.num_E = 0
