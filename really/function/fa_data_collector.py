'''
Created on 13 May 2019

@author: enerve
'''

import logging
from really import util
import numpy as np

class FADataCollector(object):
    '''
    Helps collect training/validation data rows for future FA updates
    '''


    def __init__(self, fa):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.fa = fa #TODO: not needed?

        self.reset_dataset()
        
    def reset_dataset(self):
        # Forget old steps history
        self.steps_history_state = []
        self.steps_history_action = []
        self.steps_history_target = []
        self.ireplay = None
        self.pos = self.neg = 0

    def replay_dataset(self):
        ''' Prepare to replay instead of record new data. FOR DEBUGGING ONLY. '''
        if len(self.steps_history_state) > 0:
            self.ireplay = 0
            self.pos = self.neg = 0

    def record(self, state, action, target):
        ''' Record incoming data rows '''

#         if target >= -0.9 and target <= 0.9:
#             return

        if self.ireplay is not None:
            # Replaying the same datapoints but recording new targets
            # Confirm it's an exact repeat
            old_action = self.steps_history_action[self.ireplay]
            if action != old_action:
                self.logger("Got %d vs %d", action, old_action)
                old_state = self.steps_history_state[self.ireplay]
                self.logger("    %s vs %s", state, old_state)
            self.steps_history_target[self.ireplay] = target
            self.ireplay += 1
        else:
            self.steps_history_state.append(state)
            self.steps_history_action.append(action)
            self.steps_history_target.append(target)

        if target > 0:
            self.pos += 1
        else:
            self.neg += 1
        
    def store_last_dataset(self, pref=""):
        fname = "dataset_" + pref
        SHS = np.asarray(self.steps_history_state)
        SHA = np.asarray(self.steps_history_action)
        SHT = np.asarray(self.steps_history_target)
        util.dump(SHS, fname, "S")
        util.dump(SHA, fname, "A")
        util.dump(SHT, fname, "t")
 
    def load_dataset(self, subdir, pref=""):
        fname = "dataset_" + pref
        self.logger.debug("Loading dataset from %s (%s)", fname, subdir)
        steps_history_state = [s for s in util.load(fname, subdir, suffix="S")]
        steps_history_action = [a for a in util.load(fname, subdir, suffix="A")]
        steps_history_target = [t for t in util.load(fname, subdir, suffix="t")]
        
        self.logger.debug("Filtering!")
        self. steps_history_state, self.steps_history_action, self.steps_history_target = [], [], []
        sum = 0
        absum = 0
        for S, a, t in zip(steps_history_state, steps_history_action, steps_history_target): 
            #TODO: move to coindrop
            if t < -0.899 or t > 0.899:
                #self.logger.debug("%0.2f target for action %d on:\n%s", t, a, S)
                sum += t
                absum += 1
                self.steps_history_state.append(S)
                self.steps_history_action.append(a)
                self.steps_history_target.append(t)
        self.logger.debug("%d sum, out of %d", sum, absum)
        
        
    def before_update(self, pref=""):
        #TODO: move to coindrop..
        #self.logger.debug("#pos: %d \t #neg: %d", self.pos, self.neg)
        pass

    def report_collected_dataset(self):
        SHT = np.asarray(self.steps_history_target)
        #TODO: move to coindrop
#         self.logger.debug("  +1s: %d \t -1s: %d", np.sum(SHT > 0.99),
#                           np.sum(SHT < 0.01))

    def get_data(self):
        return self.steps_history_state, self.steps_history_action, self.steps_history_target
      
      
if __name__ == '__main__':
    from really import cmd_line
    from really import log
    
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'Coindrop'

    logger = logging.getLogger()
    log.configure_logger(logger, "Coindrop")
    logger.setLevel(logging.DEBUG)
    
    dc = FADataCollector(None)
    dir = "543562_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.0005_r0.5_b512_i1500_FFEelv__NNconvnet_lookab5__"
    dc.load_dataset(dir, "final_t")
    
    
    
    