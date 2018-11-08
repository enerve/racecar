'''
Created on Nov 3, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from . import Driver
from environment import Environment
from function import PolynomialRegression
from function import QLookup
import util

class SarsaRegressionDriver(Driver):
    '''
    An agent that learns to drive a car along a track, optimizing using 
    Sarsa and a linear regression function approximator
    '''

    def __init__(self, alpha, gamma, explorate,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions):
        '''
        Constructor
        '''
        super().__init__(num_junctures,
                         num_lanes,
                         num_speeds,
                         num_directions,
                         num_steer_positions,
                         num_accel_positions)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_alg = "sarsa_regression_%d_%0.1f_%0.1f" % (explorate, alpha, gamma)
        self.logger.debug("Algorithm: %s", util.pre_alg)

        self.gamma = gamma  # weight given to predicted future
        self.explorate = explorate # Inclination to explore, e.g. 0, 10, 1000
        self.alpha = alpha  # learning rate for updating Q

        # This is temporary.
        # TODO: feed in the FA object through an init variable
        self.fa = QLookup(
                        alpha,
                        num_junctures,
                        num_lanes,
                        num_speeds,
                        num_directions,
                        num_steer_positions,
                        num_accel_positions)
        self.fa2 = PolynomialRegression(
                        0.01, # alpha ... #4e-5 old alpha without batching
                        0.02, # regularization constant
                        256, # batch_size
                        250, # max_iterations
                        num_junctures,
                        num_lanes,
                        num_speeds,
                        num_directions,
                        num_steer_positions,
                        num_accel_positions)
        self.avg_delta2 = 0
        self.stat_dlm2 = []

        
        # TODO: move N to FA?
        # N is the count of visits to state
        self.N = np.zeros((self.num_junctures,
                           self.num_lanes,
                           self.num_speeds,
                           self.num_directions), dtype=np.int32)
                           
        # C is the count of visits to state/action
        self.C = np.zeros((num_junctures,
                           num_lanes,
                           num_speeds,
                           num_directions,
                           num_steer_positions,
                           num_accel_positions), dtype=np.int32)
        # Rs is the average reward at juncture (for statistics)
        self.Rs = np.zeros((num_junctures), dtype=np.float)
        self.avg_delta = 0
        self.restarted = False

        self.stat_e_100 = []
        self.stat_qm = []
        self.stat_cm = []
        self.stat_rm = []
        
        self.stat_debug_qv = []
        self.stat_debug_n = []

        self.stat_e_200 = []
        self.q_plotter = util.Plotter("Q value at junctures along lanes")
        self.qx_plotter = util.Plotter("Max Q value at junctures along lanes")
        self.c_plotter = util.Plotter("C value at junctures along lanes")

        # track average change in Q, as iterations progress
        self.stat_dlm = []
        
        self.actions_matched = 0
        self.total_max_actions_picked = 0

    def restart_exploration(self, scale_explorate=1):
        super().restart_exploration()
        
        # N is the count of visits to state
        self.N = np.zeros((self.num_junctures,
                           self.num_lanes,
                           self.num_speeds,
                           self.num_directions), dtype=np.int32)
        self.explorate *= scale_explorate
        self.logger.debug("Restarting with explorate=%d", self.explorate)
        self.avg_delta = 0
        self.restarted = True

    def _pick_action(self, S, run_best):
        if run_best:
            epsilon = 0
        else:
            n = self.N[S]
            N0 = self.explorate
            epsilon = N0 / (N0 + n)
      
        if random.random() >= epsilon:
            # Pick best
            #self.logger.debug(S)
            steer, accel = self.fa.best_action(S)
            
            #debugging
#             steer2, accel2 = self.fa2.best_action(S)
#             if steer == steer2:
#                 self.actions_matched += 0.5
#             if accel == accel2:
#                 self.actions_matched += 0.5
#             self.total_max_actions_picked += 1 
            
            #debugging
#             if run_best:
# #                 my_s, my_a = MY_IDEAL_A[S[0]]
# #                 self.logger.debug("I prefer: %d whose Q is %0.2f or %0.2f ",
# #                                   my_s, As[my_s, 1], As[my_s, 2])
#                 self.logger.debug("  Chosen: %d, %d whose Q is %0.2f ... at %s", 
#                                   steer, accel, As[steer, accel], S)
#                 self.logger.debug("  Options: %s", As)
        else:
            r = random.randrange(
                self.num_steer_positions * self.num_accel_positions)
            steer, accel = divmod(r, self.num_accel_positions)
        
        return (steer, accel)
    

    def run_episode(self, track, car, run_best=False):
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=run_best)
        S = environment.state_encoding()
        total_R = 0
        
        A_ = None
        A = self._pick_action(S, run_best)
        while S is not None:
            I = S + A

            R, S_ = environment.step(A)
            
            Q_at_next = 0
            if S_ is not None:
                A_ = self._pick_action(S_, run_best)
                Q_at_next = self.fa.value(S_, A_)

            target = R + self.gamma * Q_at_next
            
            self.fa.record(S, A, target)
            
            # (TEMP) Also train fa2, to compare for debugging
            self.fa2.record(S, A, target)
            
            self.C[I] += 1
            self.N[S] += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            if True:# self.C[I] > 1:
                delta = (target - self.fa.value(S, A))
                self.avg_delta += 0.02 * (delta - self.avg_delta)
                delta2 = (target - self.fa2.value(S, A))
                self.avg_delta2 += 0.02 * (delta2 - self.avg_delta2)

            S, A = S_, A_
            total_R += R
            
        return total_R, environment
    
    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)
        
        if util.checkpoint_reached(ep, num_episodes // 100):
            self.stat_e_100.append(ep)

            self.stat_dlm.append(self.avg_delta)
            self.stat_dlm2.append(self.avg_delta2)
            
            #self.logger.debug("Portion of matched actions: %0.4f",
            #                  self.actions_matched / self.total_max_actions_picked)

    def learn_from_history(self):
        self.fa2.update()

        # debugging
        self.actions_matched = 0
        self.total_max_actions_picked = 0

    def report_stats(self, pref):
        super().report_stats(pref)
        
        self.fa2.report_stats(pref)

        self.logger.debug("Final W: %s", self.fa2.W)
        util.plot([self.stat_dlm, self.stat_dlm2], self.stat_e_100,
                  ["Avg ΔQ table", "Avg ΔQ LinReg"], pref="delta",
                  ylim=None)#(-100, 1000))
