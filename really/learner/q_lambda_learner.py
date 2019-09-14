'''
Created on Apr 29, 2019

@author: enerve
'''

import logging

from really.learner import Learner

class QLambdaLearner(Learner):
    '''
    An observer that uses Q(λ) to generate training data from episodes
    '''

    def __init__(self,
                 config,
                 lam,
                 gamma,
                 fa):
         
        super().__init__(config, fa)
 
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.lam = lam      # lambda lookahead parameter
        self.gamma = gamma  # weight given to future rewards

        self.eligible_mult = [(lam * gamma) ** i for i in range(self.max_moves)]        

    def prefix(self):
        pref = "q_lambda_g%s_l%0.2f" % (self.gamma, self.lam) + self.fa.prefix()
        return pref

    def _process_steps(self, steps_history, data_collector):
        ''' Extracts training data from the given episode '''
        S, A = None, None
        max_A, Q_at_max_next = None, 0
        num_E = 0
        self.eligible_states = [None for i in range(self.max_moves)]
        self.eligible_state_target = [0 for i in range(self.max_moves)]
        for i, (R_, S_, A_) in enumerate(steps_history):
            if i > 0:
                if A == max_A:
                    # Reuse fa value from the previous iteration.
                    curr_value = Q_at_max_next
                else:
                    # Unable to reuse. Need to calculate fa value.
                    curr_value = self.fa.value(S, A)

                if S_ is not None:
                    # off-policy
                    # TODO: speed this up, or parallelize it
                    max_A, Q_at_max_next, _ = self.fa.best_action(S_)
                    #Q_at_max_next = self.fa.value(S_, max_A)
                else:
                    max_A, Q_at_max_next = None, 0

                # Learn from the reward gotten for action taken last time
                target = R_ + self.gamma * Q_at_max_next
                
                self.eligible_states[num_E] = (S, A)
                self.eligible_state_target[num_E] = curr_value
                delta = (target - curr_value)
                
                num_E += 1
                for j in range(num_E):
                    self.eligible_state_target[j] += delta * self.eligible_mult[num_E-j-1]
                if A_ != max_A:
                    # The policy diverted from Q* policy so restart eligibilities
                    # But first, flush the eligibility updates into the FA
                    self._record_eligibles(num_E, data_collector)
                    num_E = 0

                self.currs.append(curr_value)
                self.targets.append(target)
                self.deltas.append(delta)
                #if R_ > 0: self.num_wins += 1 
                #if R_ < 0: self.num_loss += 1
                
            S, A = S_, A_
            
        self._record_eligibles(num_E, data_collector)

    def _record_eligibles(self, num_E, data_collector):
        for i in range(num_E):
            S, A = self.eligible_states[i]
            target = self.eligible_state_target[i]
            data_collector.record(S, A, target)
