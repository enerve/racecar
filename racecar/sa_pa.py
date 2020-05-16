'''
Created on 21 Sep 2019

@author: enerve
'''

import collections
import logging
import random
import torch
import torch.nn as nn

from really import util
from really.function import AllSequential, GivenGradient

class SA_PA():  #TODO: should this in fact extend from ES?
    '''
    '''

    def __init__(self,
                 config,
                 model,
                 feature_eng):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Racecar SA FA")

        # actions
        self.num_steer_positions = config.NUM_STEER_POSITIONS
        self.num_accel_positions = config.NUM_ACCEL_POSITIONS
        self.valid_actions = [(s, a) for s in range(self.num_steer_positions)
                              for a in range(self.num_accel_positions)]

        self.model = model
        self.feature_eng = feature_eng
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')
        
        
    def prefix(self):
        return 'racecar_sa_M%s_F%s' % (self.model.prefix(),
                                    self.feature_eng.prefix())

    # ------- Model init --------

    def init_default_model(self):
        A = 2000
        B = 2000
        C = 2000
        net = AllSequential(collections.OrderedDict([
            ('1lin', nn.Linear(self.feature_eng.num_inputs, A)),
            ('1relu', nn.LeakyReLU()),
            ('1bn', nn.BatchNorm1d(A)),
            ('2ln', nn.Linear(A, B)),
            ('2relu', nn.LeakyReLU()),
            ('2bn', nn.BatchNorm1d(B)),
    #         nn.Linear(h2, h3),
    #         nn.Sigmoid(),
            #('3lin', nn.Linear(C, self.num_outputs())),
            ('given', GivenGradient(C)),
            ]))
    
        self.model.init_net(net)

    # ------- Running --------

#     def _value(self, state, action):
#         X = self.feature_eng.x_adjust(state, action)
#         output = self.model.value(X.unsqueeze(0))[0]
#         return output
#     
#     def value(self, state, action):
#         output = self._value(state, action)
#         return output[0].item()

    def values(self, state):
        x_list = [self.feature_eng.x_adjust(state, a) for a in self.valid_actions]
        Xb = torch.stack(x_list).to(self.device)
        output, activations = self.model.all_values(Xb)
        return output, activations, self.valid_actions
    
#     def encoding_vectors(self, state):
#         _, activations, _ = self.values(state)
#         return activations['2bn']

    def best_action(self, state):
        V = self.values(state)
        i = torch.argmax(V).item()
        v = V[i].item()
        return self.action_from_index(i), v, V.tolist()

#     def random_action(self, state):
#         r = random.randrange(
#             self.num_steer_positions * self.num_accel_positions)
#         return self.action_from_index(r)

    def pick_action(self, state):
        vals, actions = self.values(state)
        
        # TODO: this is softmax specific? move to policy learner??
        # TODO: is this logarithmic, should we exponentiate first?
        action_probs = vals
        
        r = random.random()
        p = 0
        for a, prob in zip(actions, action_probs):
            p += prob
            if p > r:
                picked_a = a
                break
        else:
            picked_a = self.valid_actions[-1]

        return picked_a, action_probs

    def num_outputs(self):
        return 1
    
    def a_index(self, a_tuple):
        steer, accel = a_tuple
        return self.num_accel_positions * steer + accel

    def action_from_index(self, a_index):
        return (a_index // self.num_accel_positions,
                a_index % self.num_accel_positions)


    # ------- Training --------

    def update(self, training_data_collector, validation_data_collector):
        ''' Updates the value function model based on data collected since
            the last update '''

        training_data_collector.before_update()

        self.train(training_data_collector, validation_data_collector)
        self.test()

    def _prepare_data(self, steps_history_state, steps_history_action,
                      steps_history_gradient):
        steps_history_x = []
        steps_history_g = []
        steps_history_mask = []

        #Sdict = {}
        #count_conflict = 0
        self.logger.debug("  Preparing for %d items", len(steps_history_state))

        teye = torch.ones(1).to(self.device)
        
        for i, (S, a, g) in enumerate(zip(
                        steps_history_state,
                        steps_history_action,
                        steps_history_gradient)):
            if i == 250000:
                self.logger.warning("------ too much to prepare ----------")
                break

            x = self.feature_eng.x_adjust(S, a)
            m = teye.clone() # Not ideal but allows flexibility in nn model

            steps_history_x.append(x)
            steps_history_g.append(g)
            steps_history_mask.append(m)

            if (i+1) % 10000 == 0:
                self.logger.debug("prepared %d", i+1)
                #self.logger.debug("  conflict count: %d" % count_conflict)
            
        #util.hist(list(Sdict.values()), bins=100, range=(2,50))
        
        return steps_history_x, steps_history_g, steps_history_mask


    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def train(self, training_data_collector, validation_data_collector):
        self.logger.debug("Preparing training data--")
        steps_history_x, steps_history_g, steps_history_m = \
            self._prepare_data(*training_data_collector.get_data())
        self.logger.debug("Preparing validation data--")
        val_steps_history_x, val_steps_history_g, val_steps_history_m = \
            self._prepare_data(*validation_data_collector.get_data())
        
        SHX = torch.stack(steps_history_x)
        SHG = torch.stack(steps_history_g)
        SHM = torch.stack(steps_history_m)
        VSHX = torch.stack(val_steps_history_x)
        VSHG = torch.stack(val_steps_history_g)
        VSHM = torch.stack(val_steps_history_m)

        self.model.train(SHX, SHG, SHM, VSHX, VSHG, VSHM)

    def test(self):
        pass

    def collect_stats(self, ep):
        self.model.collect_stats(ep)
    
    def collect_epoch_stats(self, epoch):
        self.model.collect_epoch_stats(epoch)
        self.feature_eng.collect_epoch_stats(epoch)
    
    def save_stats(self, pref=""):
        self.model.save_stats(pref)

    def load_stats(self, subdir, pref=""):
        self.model.load_stats(pref)

    def report_stats(self, pref=""):
        self.model.report_stats(pref)
    
    def live_stats(self):
        self.model.live_stats()
        
    def save_model(self, pref=""):
        self.model.save_model(pref)

    def load_model(self, load_subdir, pref=""):
        self.model.load_model(load_subdir, pref)
