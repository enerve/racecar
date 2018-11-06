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
                        num_junctures,
                        num_lanes,
                        num_speeds,
                        num_directions,
                        num_steer_positions,
                        num_accel_positions)
        self.fa2 = PolynomialRegression(
                        4e-5,  # alpha
                        100, # regularization constant
                        num_junctures,
                        num_lanes,
                        num_speeds,
                        num_directions,
                        num_steer_positions,
                        num_accel_positions)
        self.avg_delta2 = 0
        self.avg_error_cost = 0
        self.avg_reg_cost = 0
        self.avg_W = 0
        self.stat_dlm2 = []
        self.stat_error_cost = []
        self.stat_reg_cost = []
        self.stat_W = []

        
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
            
            delta = (target - self.fa.value(S, A))
            self.fa.update(S, A, self.alpha, delta)
            
            # (TEMP) Also train fa2, to compare for debugging
            delta2 = (target - self.fa2.value(S, A))
            error_cost, reg_cost, sumW, W = self.fa2.update(S, A, delta2)
            
            self.C[I] += 1
            self.N[S] += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            if True:# self.C[I] > 1:
                self.avg_delta += 0.02 * (delta - self.avg_delta)
                self.avg_delta2 += 0.02 * (delta2 - self.avg_delta2)
                self.avg_error_cost += 0.02 * (error_cost - self.avg_error_cost)
                self.avg_reg_cost += 0.2 * (reg_cost - self.avg_reg_cost)
                self.avg_W += 0.02 * (W - self.avg_W)

            S, A = S_, A_
            total_R += R
            
        return total_R, environment

    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)
        
        if True:#util.checkpoint_reached(ep, num_episodes // 100):
            self.stat_e_100.append(ep)

            self.stat_dlm.append(self.avg_delta)
            self.stat_dlm2.append(self.avg_delta2)
            self.stat_error_cost.append(self.avg_error_cost)
            self.stat_reg_cost.append(self.avg_reg_cost)
            self.stat_W.append(np.copy(self.avg_W))

    def report_stats(self, pref):
        super().report_stats(pref)

        self.logger.debug("Final W: %s", self.fa2.W)
        util.plot([self.stat_dlm, self.stat_dlm2], self.stat_e_100,
                  ["Avg ΔQ table", "Avg ΔQ LinReg"], pref="delta",
                  ylim=(-100, 1000))
#         util.plot([self.stat_error_cost, self.stat_reg_cost], self.stat_e_100,
#                   ["Avg error cost", "Avg regularization cost"], pref="cost",
#                   ylim=(0, 10000))
        util.plot([self.stat_error_cost], self.stat_e_100,
                  ["Avg error cost"], pref="cost",
                  ylim=(0, 50000))
        
        sW = np.asarray(self.stat_W).T
        labels = ["W[%d]" % x for x in range(len(sW))]
        util.plot(sW, self.stat_e_100, labels, pref="W", ylim=(-10, 10))
