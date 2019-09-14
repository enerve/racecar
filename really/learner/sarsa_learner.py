'''
Created on May 22, 2019

@author: enerve
'''

import logging

from really.learner import Learner

class SarsaLearner(Learner):
    '''
    An observer that uses Sarsa to generate training data from episodes
    '''

    def __init__(self,
                 config,
                 gamma,
                 fa):
        super().__init__(config, fa)
 
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.gamma = gamma  # weight given to future rewards

    def prefix(self):
        pref = "sarsa_g%s_" % self.gamma + self.fa.prefix()
        return pref

    def _process_steps(self, steps_history, data_collector):
        ''' Extracts training data from the given episode '''
        S, A = None, None
        Q_at_next = 0
        for i, (R_, S_, A_) in enumerate(steps_history):
            if i > 0:
                if i > 1:
                    # Reuse fa value from the previous iteration.
                    curr_value = Q_at_next
                else:
                    # Unable to reuse. Need to calculate fa value.
                    curr_value = self.fa.value(S, A)

                if S_ is not None:
                    # TODO: speed this up, or parallelize it
                    Q_at_next = self.fa.value(S_, A_)
                else:
                    Q_at_next = 0

                # Learn from the reward gotten for action taken last time
                target = R_ + self.gamma * Q_at_next
                data_collector.record(S, A, target)

                delta = (target - curr_value)

                self.currs.append(curr_value)
                self.targets.append(target)
                self.deltas.append(delta)
                #if R_ > 0: self.num_wins += 1 
                #if R_ < 0: self.num_loss += 1

            S, A = S_, A_

