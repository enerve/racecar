'''
Created on 13 May 2019

@author: enerve
'''

import collections
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random

from .value_function import ValueFunction
from really import util
from .conv_net import AllSequential, Flatten

class NN_Bound_FA(ValueFunction):
    '''
    A neural-network action-value function approximator for a single (bound)
    action.
    "Bound actions": the action is first applied to the board state before 
    feeding into the NN, and the output is a single value node.
    '''

    def __init__(self,
                 alpha,
                 regularization_param,
                 batch_size,
                 max_iterations,
                 model):
        '''
        Constructor
        '''

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.logger.debug("Using NN Bound FA")

        self.alpha = alpha
        self.regularization_param = regularization_param
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.model = model
        
        self.device = torch.device('cuda' if util.use_gpu else 'cpu')

        self.num_outputs = 1

        # Stats / debugging
        self.stat_error_cost = []
        self.stat_reg_cost = []
        self.stat_val_error_cost = []
        self.stat_activations = []
        self.W_norm = []
                
        self.sids = self._sample_ids(3000, self.batch_size)
        self.last_loss = torch.zeros(self.batch_size, 7).cuda()
                
    def initialize_default_net(self):
        self.logger.debug("Creating new model")
        A = 100
        B = 300
        C = 100
        D = 50
        E = 25
        net = AllSequential(collections.OrderedDict([
            ('1conv', nn.Conv2d(2, A, kernel_size=4, stride=1, padding=1)),
            #nn.Dropout2d(p=0.2),
            ('1relu', nn.LeakyReLU()),
            ('1bn', nn.BatchNorm2d(A)),
            ('2conv', nn.Conv2d(A, B, kernel_size=2, stride=1, padding=0)),
            #nn.Dropout2d(p=0.5),
            ('2relu', nn.LeakyReLU()),
            ('2bn', nn.BatchNorm2d(B)),
            ('3flatten', Flatten()),
            ('3lin', nn.Linear(B*5*4, C)),
            #nn.Dropout(p=0.5),
            ('3relu', nn.LeakyReLU()),
            #('3bn', nn.BatchNorm1d(C)),
            ('4lin', nn.Linear(C, D)),
            #nn.Dropout(p=0.2),
            ('4relu', nn.LeakyReLU()),
            #('4bn', nn.BatchNorm1d(D)),
            ('5lin', nn.Linear(D, E)),
            #nn.Dropout(p=0.2),
            ('5relu', nn.LeakyReLU()),
            #('5bn', nn.BatchNorm1d(E)),
            ('6lin', nn.Linear(E, 1)),
            #('6sigmoid', nn.Sigmoid())
            #nn.Tanh()
            ]))

#         net = nn.Sequential(
#             nn.Conv2d(2, 50, kernel_size=3, stride=1, padding=0),
#             nn.Dropout2d(p=0.2),
#             nn.ReLU(),
#             nn.Conv2d(50, 80, kernel_size=2, stride=1, padding=0),
#             nn.Dropout2d(p=0.2),
#             nn.ReLU(),
#             nn.Conv2d(80, 100, kernel_size=2, stride=1, padding=0),
#             nn.Dropout2d(p=0.2),
#             nn.ReLU(),
#             nn.Conv2d(100, 200, kernel_size=2, stride=1, padding=0),
#             nn.Dropout2d(p=0.2),
#             nn.ReLU(),
#             Flatten(),
#             nn.Linear(200*2*1, 500),
#             nn.Dropout(p=0.2),
#             nn.ReLU(),
#             nn.Linear(500, 1),
#             nn.Dropout(p=0.2),
#             nn.Sigmoid())

#         net = nn.Sequential(
#             nn.Conv2d(2, 100, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             Flatten(),
#             nn.Linear(100*7*6, 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, 1),
#             nn.Sigmoid())

        self.init_net(net)

    def _adjust_output(self, out):
        return out
        #return out * 2 - 1
        
    def _adjust_target(self, t):
        return t
        #return (t+1.0) / 2

    def init_net(self, net):        
        net.cuda(self.device)

        self.criterion = nn.MSELoss(reduce=False)
        #self.criterion = nn.SmoothL1Loss(reduce=False)
        #self.criterion = nn.BCELoss(reduce=False)
        #self.criterion = nn.BCEWithLogitsLoss(reduce=False)
        
        self.optimizer = optim.Adam(
            net.parameters(),
            lr=self.alpha,
            weight_decay=self.regularization_param,
            amsgrad=False)

        self.net = net
        
        self.logger.debug("Net:\n%s", self.net)
        self.logger.debug("Criterion: %s", self.criterion)
        
    def prefix(self):
        return 'neural_bound_a%s_r%s_b%d_i%d_F%s_NN%s' % (self.alpha, 
                                     self.regularization_param,
                                     self.batch_size,
                                     self.max_iterations,
                                     self.model.prefix(),
                                     "convnet")

    # ---------------- Run phase -------------------

    def _bind_action(self, S, action):
        ''' Applies the action to the state board and returns result '''
        B = np.copy(S)
        for h in range(6):
            if B[h, action] == 0:
                B[h, action] = 1
                return B
        
        self.logger.warning("Action on full column! %d on \n%s", action, S)
        return None        
        
#     def _value(self, S, actions):
#         ''' Gets values for all given actions '''
#         x_list = []
#         for action in actions:
#             B = self._bind_action(S, action)
#             x = self.model.feature(B)
#             x_list.append(x)
# 
#         # Sends all bound actions as a batch to NN
#         with torch.no_grad():
#             XB = torch.stack(x_list).to(self.device)
#             output = torch.t(self.net(XB))
#         return self._adjust_output(output[0])

    def _bound_value(self, x_list):
        l = len(x_list) // 2
        # Sends all bound actions as a batch to NN
        with torch.no_grad():
            XB = torch.stack(x_list).to(self.device)
            output = torch.t(self.net(XB)[-1])
            op1 = output[0][0:l]
            op2 = output[0][l:2*l]
            opavg = (op1+op2)/2
        return self._adjust_output(opavg)

    def bound_value(self, B):
        x_list = []
        x_list.append(self.model.feature(B))
        x_list.append(self.model.feature(np.flip(B, axis=1).copy()))
        return self._bound_value(x_list)

    def _value(self, S, actions):
        ''' Gets values for all given actions '''
        x_list = []
        xm_list = []
        for action in actions:
            B = self._bind_action(S, action)
            x_list.append(self.model.feature(B))
            xm_list.append(self.model.feature(np.flip(B, axis=1).copy()))
        x_list.extend(xm_list)

        return self._bound_value(x_list)

    def value(self, S, action):
        output = self._value(S, [action])
        return output[0].item()

    # TODO: move to coindrop
    def _valid_actions(self, S):
        return np.nonzero(S[6-1] == 0)[0]

    def best_action(self, S):
        actions_list = self._valid_actions(S)
        with torch.no_grad():
            V = self._value(S, actions_list)
            i = torch.argmax(V).item()
            v = V[i].item()
        return actions_list[i], v, V.tolist()

    # TODO: maybe just move this to EexplorationStrategy
    def random_action(self, S):
        actions_list = self._valid_actions(S)
        return random.choice(actions_list)
      
    # ---------------- Update phase -------------------
    
    def update(self, training_data_collector, validation_data_collector):
        ''' Updates the value function model based on data collected since
            the last update '''

        training_data_collector.before_update()

        self.train(training_data_collector, validation_data_collector)
        self.test()

    def _prepare_data(self, steps_history_state, steps_history_action,
                      steps_history_target):
        steps_history_x = []
        steps_history_t = []
        teye = torch.eye(self.num_outputs).to(self.device)

        self.logger.debug("  Preparing for %d items", len(steps_history_state))
        
        for i, (S, a, t) in enumerate(zip(
                        steps_history_state,
                        steps_history_action,
                        steps_history_target)):
            if i == 250000:
                self.logger.warning("------ too much to prepare ----------")
                break
            
            t = self._adjust_target(t)
            B = self._bind_action(S, a)
#             if t < 0.001 or t > 0.999:
#                 self.logger.debug("%0.2f target for action %d on bound:\n%s", t, a, B)
            for flip in [False, True]:
                if flip: B = np.flip(B, axis=1).copy()
                
                x = self.model.feature(B)
                
                steps_history_x.append(x)
                steps_history_t.append(t)

            if (i+1) % 10000 == 0:
                self.logger.debug("prepared %d*2", i+1)                
            
        return steps_history_x, steps_history_t

    def _sample_ids(self, l, n):
        ids = torch.cuda.FloatTensor(n) if util.use_gpu else torch.FloatTensor(n)
        ids = (l * ids.uniform_()).long()
        return ids

    def train(self, training_data_collector, validation_data_collector):
        self.logger.debug("Preparing training data--")
        steps_history_x, steps_history_t = \
            self._prepare_data(*training_data_collector.get_data())
        self.logger.debug("Preparing validation data--")
        val_steps_history_x, val_steps_history_t = \
            self._prepare_data(*validation_data_collector.get_data())
        
        SHX = torch.stack(steps_history_x).to(self.device)
        SHT = torch.tensor(steps_history_t).to(self.device)
        VSHX = torch.stack(val_steps_history_x).to(self.device)
        VSHT = torch.tensor(val_steps_history_t).to(self.device)
        
        self.logger.debug("Training with %d items...", len(steps_history_x))

#         self.logger.debug("  +1s: %d \t -1s: %d", torch.sum(SHT > 0.99),
#                           torch.sum(SHT < 0.01))


        N = len(steps_history_t)
        
        # for stats
        preferred_samples = 1000
        period = self.max_iterations // preferred_samples
        period = max(period, 10)
        if period > self.max_iterations:
            self.logger.warning("max_iterations too small for period plotting")

        sum_error_cost = torch.zeros(self.num_outputs).to(self.device)
        sum_error_cost.detach()
        count_actions = torch.zeros(self.num_outputs).to(self.device)

        for i in range(self.max_iterations):
            self.optimizer.zero_grad()
            
            if self.batch_size == 0:
                # Do full-batch
                X = SHX   # N x di
                Y = SHT   # N
            else:
                ids = self._sample_ids(N, self.batch_size)
                X = SHX[ids]   # b x di
                Y = SHT[ids]   # b
            Y = torch.unsqueeze(Y, 1)   # b x 1
            
            # forward
            outputs = self.net(X)[-1]       # b x 1
            # loss
            loss = self.criterion(outputs, Y)  # b x 1
            # backward
            onez = torch.ones(loss.shape).to(self.device) #TODO: move out?
            loss.backward(onez)
            
            # updated weights
            self.optimizer.step()
            
            # Stats
            with torch.no_grad():
                suml = torch.sum(loss, 0)
                countl = torch.sum(loss > 0, 0).float()

                sum_error_cost.add_(suml)  # 1
                count_actions.add_(countl)  # 1

                if (i+1) % period == 0:
                    mean_error_cost = sum_error_cost / (count_actions + 0.01)
                    self.stat_error_cost.append(mean_error_cost.cpu().numpy())
    
                    #self.logger.debug("  loss=%0.2f", sum_error_cost.mean().item())
    
                    torch.zeros(self.num_outputs, out=sum_error_cost)
                    torch.zeros(self.num_outputs, out=count_actions)
                    
                    # Validation
                    X = VSHX
                    Y = torch.unsqueeze(VSHT, 1)
                    outputs = self.net(X)
                    op_Y = outputs[-1]       # b x 1
                    loss = self.criterion(op_Y, Y)  # b x 1
                    
                    suml = torch.sum(loss, 0)
                    countl = torch.sum(loss > 0, 0).float()
                    mean_error_cost = suml / (countl + 0.01)
                    self.stat_val_error_cost.append(mean_error_cost.cpu().numpy())
                    
                    Yo = (op_Y > 0).float()# - (op_Y < 0.5).float()
                    Y = (Y > 0).float()
                    n_o = torch.sum(Y == Yo)
                    self.logger.debug(" validation target -ish: %d / %d \t cost %0.2f",
                                      n_o, Yo.shape[0], self.stat_val_error_cost[-1])
                    val_o = op_Y

                    # Dead activations (ReLUs e.g.)
                    act_list = []
                    for op in outputs:
                        activations_on = (op >= 0.000000000001) # b x a
                        #node_activations = torch.sum(activations_on, 0) > 0 # a
                        ratio_alive = torch.mean(activations_on.float()).cpu()
                        act_list.append(ratio_alive)
                    self.stat_activations.append(act_list)

                    # Weight parameters
#                     w_norm_list = []
#                     for param in self.net.parameters():
#                         w_norm_list.append(torch.norm(param.data))                        
                    self.W_norm.append(
                        [torch.norm(param.data).item() for param in self.net.parameters()])

                    #self.live_stats()

            if (i+1) % 1000 == 0:
                self.logger.debug("   %d / %d", i+1, self.max_iterations)

        self.logger.debug("  trained \tN=%s \tE=%0.3f \tVE=%0.3f", N,
                          self.stat_error_cost[-1].mean().item(),
                          self.stat_val_error_cost[-1].mean().item())

        for param in self.net.parameters():
            self.logger.debug("  W=%0.6f dW=%0.6f    %s", 
                              torch.mean(torch.abs(param.data)),
                              torch.mean(torch.abs(param.grad)),
                              param.shape)

        if False:
            # Log the worst predictions, for debugging
            Y = torch.unsqueeze(VSHT, 1)
            d = torch.abs(Y - val_o)
            slist = validation_data_collector.get_data()[0]
            alist = validation_data_collector.get_data()[1]
            for _ in range(10):
                i = torch.argmax(d).item()
                self.logger.debug(" Val: %0.2f instead of %0.2f for action %d on\n%s",
                                  val_o[i], Y[i], alist[i//2], slist[i//2])#, VSHX[i])
                d[i] = 0

        

    def test(self):
        pass

    def collect_stats(self, ep):
        pass
    
    def _cost_arrays_numpy(self, skip=0):
        n_cost = np.asarray(self.stat_error_cost[skip:]).T
        labels = list(range(n_cost.shape[1]))        

        n_v_cost = np.asarray(self.stat_val_error_cost[skip:]).T
        labels.extend(["val%d" % i for i in range(n_v_cost.shape[1])])
        cost = np.concatenate([n_cost, n_v_cost], axis=0)
        
        return cost, n_cost, n_v_cost
    
    def save_stats(self, pref=""):
        cost, n_cost, n_v_cost = self._cost_arrays_numpy(skip=0)

        util.dump(cost, "statsNNcost", pref)

    def load_stats(self, subdir, pref=""):
        cost = util.load("statsNNcost", subdir, pref)
        n = len(cost)
        
        n_cost, n_v_cost = cost[0:n//2], cost[n//2:n]
        self.stat_error_cost = [x for x in n_cost.T]
        self.stat_val_error_cost = [x for x in n_v_cost.T]
        

    def report_stats(self, pref=""):
        num = len(self.stat_error_cost[1:])

        cost, n_cost, n_v_cost = self._cost_arrays_numpy(skip=1)
        labels = list(range(n_cost.shape[1]))        
        labels.extend(["val%d" % i for i in range(n_v_cost.shape[1])])
        util.plot(cost,
                  range(num),
                  labels = labels,
                  title = "NN training/validation cost across actions", #TODO: not across
                  pref=pref+"cost",
                  ylim=None)


#         avgcost = np.stack([n_cost.mean(axis=0), n_v_cost.mean(axis=0)], axis=0)
#         util.plot(avgcost,
#                   range(num),
#                   labels = ["training cost", "validation cost"],
#                   title = "NN training/validation cost",
#                   pref=pref+"avgcost",
#                   ylim=None)

        W = np.asarray(self.W_norm).T

        #labels = ["%s"%c for c in self.net.modules()][1:]
        
        util.plot(W,
                  range(len(self.W_norm)),
                  #labels=labels,
                  labels=list(range(len(self.W_norm))),
                  title="Weights L2 norm",
                  pref=pref+"W")

        A = np.asarray(self.stat_activations).T
        util.plot(A,
                  range(len(self.stat_activations)),
                  #labels=labels,
                  labels=self.net.get_names(),
                  title="Ratio of nodes alive",
                  pref=pref+"A")

        relu_list = []
        for key, acts in zip(self.net.get_names(), A):
            if 'relu' in key:
                relu_list.append(acts)
        Ar = np.asarray(relu_list)
        util.plot(Ar,
                  range(len(self.stat_activations)),
                  #labels=labels,
                  labels=range(len(relu_list)),
                  title="Ratio of ReLU nodes alive",
                  pref=pref+"Ar")

    def live_stats(self):
        num = len(self.stat_error_cost[1:])
        
        if num < 1:
            return

        n_cost = np.asarray(self.stat_error_cost[1:]).T
        n_v_cost = np.asarray(self.stat_val_error_cost[1:]).T
        avgcost =  np.stack([n_cost.mean(axis=0), n_v_cost.mean(axis=0)], axis=0)

        util.plot(avgcost,
                  range(num),
                  labels = ["training cost", "validation cost"],
                  title = "NN training/validation cost",
                  live=True)
        
    def save_model(self, pref=""):
        self.logger.debug("Saving model")
        util.torch_save(self.net, "boundNN_" + pref)
        self.logger.debug("Saving model state dict")
        util.torch_save(self.net.state_dict(), "boundNN_sd_" + pref)

    def load_model(self, load_subdir, pref=""):
        fname = "boundNN_" + pref
        self.logger.debug("Loading model %s", fname)
        net = util.torch_load(fname, load_subdir)
        net.eval()
        self.init_net(net)

    def load_model_params(self, load_subdir, pref=""):
        ''' Called to load NN weights. Assumes default NN has been initialized '''
        fname = "boundNN_sd_" + pref
        self.logger.debug("Loading model %s", fname)
        self.net.load_state_dict(util.torch_load(fname, load_subdir))
        self.net.eval()

    def export_to_onnx(self, fname):
        dummy_state = np.random.randint(-1, 2, (6, 7))
        with torch.no_grad():
            dummy_X = self.model.feature(dummy_state).unsqueeze(0)
            util.torch_export(self.net, dummy_X, fname)
        self.logger.debug("Exported to ONNX")
        
    def viz(self):
        from torchviz import make_dot
        dummy_state = np.random.randint(-1, 2, (6, 7))
        dummy_X = self.model.feature(dummy_state).unsqueeze(0)
        out = self.net(dummy_X)[-1]
        dot = make_dot(out, params=dict(self.net.named_parameters()))
        dot.format = 'svg'
        dot.render()
        self.logger.debug("plotted viz")
