'''
Created on Sep 6, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from driver import Driver
from environment import Environment
import util

class QDriver(Driver):
    '''
    An agent that learns to drive a car along a track, optimizing using 
    Q-learning
    '''

    def __init__(self, alpha, gamma, explorate,
                 num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions,
                 load_Q_filename=None,
                 load_N_filename=None):
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
    
        util.pre_alg = "qlearn_%d_%0.1f_%0.1f" % (explorate, alpha, gamma)

        self.alpha = alpha  # learning rate for updating Q
        self.gamma = gamma  # weight given to predicted future
        self.explorate = explorate # Inclination to explore, e.g. 0, 10, 1000
        
        # Q is the learned value of a state/action
        if load_Q_filename is not None:
            self.logger.debug("Loading Q from: %s", load_Q_filename)
            stored_Q = util.load(load_Q_filename)
            self.Q = stored_Q.reshape((num_junctures,
                                       num_lanes,
                                       num_speeds,
                                       num_directions,
                                       num_steer_positions,
                                       num_accel_positions))
        else:
            self.Q = np.zeros((num_junctures,
                               num_lanes,
                               num_speeds,
                               num_directions,
                               num_steer_positions,
                               num_accel_positions))

        # N is the count of visits to state
        if load_N_filename is not None:
            self.logger.debug("Loading N from: %s", load_N_filename)
            stored_N = util.load(load_N_filename)
            self.N = stored_N.reshape((num_junctures,
                                       num_lanes,
                                       num_speeds,
                                       num_directions))
        else:
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

        # track max Q per juncture, as iterations progress
        self.stat_juncture_maxQ = []
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

    def pick_action(self, S, run_best):
        if run_best:
            epsilon = 0
        else:
            n = self.N[S]
            N0 = self.explorate
            epsilon = N0 / (N0 + n)
      
        if random.random() >= epsilon:
            # Pick best
            As = self.Q[S]
            steer, accel = np.unravel_index(np.argmax(As, axis=None), As.shape)
            
            #debugging
            #if run_best:
            #    my_s, my_a = MY_IDEAL_A[S[0]]
            #    self.logger.debug("I prefer: %d whose Q is %0.2f or %0.2f ",
            #                      my_s, As[my_s, 1], As[my_s, 2])
            #    self.logger.debug("  Chosen: %d, %d whose Q is %0.2f ... at %s", 
            #                      steer, accel, As[steer, accel], S)
        else:
            r = random.randrange(
                self.num_steer_positions * self.num_accel_positions)
            steer, accel = divmod(r, self.num_accel_positions)
        
        return (steer, accel)
    
    def max_at_state(self, S):
        return 0 if S is None else np.max(self.Q[S])
        
    def q_index(self, state, action):
        m, l, v, d = state
        s, a = action
        return (m, l, v, d, s, a)

    def run_episode(self, track, car, run_best=False):
        environment = Environment(track,
                                  car,
                                  self.num_junctures,
                                  should_record=run_best)
        S = environment.state_encoding() # 0 2 0 0
        total_R = 0
        
        while S is not None:
            A = self.pick_action(S, run_best)
            
            I = self.q_index(S, A)

            R, S_ = environment.step(A)

            target = R + self.gamma * self.max_at_state(S_)
            delta = (target - self.Q[I])
            self.Q[I] += self.alpha * delta
            self.C[I] += 1
            self.N[S] += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            if self.C[I] > 1:
                self.avg_delta += 0.02 * (delta - self.avg_delta)

            S = S_
            total_R += R
            
        return total_R, environment
            
    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)
        
        if ep > 0 and ep % (num_episodes // 100) == 0:
            self.stat_e_100.append(ep)

            self.stat_qm.append(self.Q.sum(axis=(1,2,3,4,5)))
            self.stat_cm.append((self.C/(self.C+1)).sum(axis=(1,2,3,4,5)))
            self.stat_rm.append(self.Rs.copy())

            #av = driver.Q[3, 1, 2, 2, 1, 2]  # what is chosen best in the end
            #bv = driver.Q[3, 1, 2, 2, 0, 1]  # what i hoped for
            #cv = driver.Q[3, 1, 2, 2, 0, 2]  #     or this
            #stat_debug_qv.append([av, bv, cv])

            #an = driver.N[3, 1, 2, 2]  # what is chosen best in the end
            #bn = driver.C[3, 1, 2, 2, 0, 1]  # what i hoped for
            #cn = driver.C[3, 1, 2, 2, 0, 2]  #     or this
            #an = driver.explorate / (driver.explorate + an)
            #stat_debug_n.append([an])

            self.stat_juncture_maxQ.append(np.max(self.Q, axis=(1,2,3,4,5)))
            self.stat_dlm.append(self.avg_delta)

        if ep > 0 and ep % (num_episodes // 200) == 0:
            self.stat_e_200.append(ep)
    
            self.q_plotter.add_image(self.plottable(self.Q, (2, 3, 4, 5)))
            self.qx_plotter.add_image(self.plottable(self.Q, (2, 3, 4, 5), pick_max=True))
            self.c_plotter.add_image(self.plottable(self.C, (2, 3, 4, 5)))
        
        #         if ep > 0 and ep % 10000 == 0:
        #             logger.debug("Done %d episodes", ep)
        #             #driver.plotQ(28, (1, 2, 3), "steering, accel") 
        #             #driver.plotQ(30, (1, 4, 5), "speed, direction")
        #             driver.plotQ(28, (2, 3, 4, 5), "lanes") 
        #                 
        #                 lines = []
        #                 labels = []
        #                 for i in range(max(0, best_MS - 9), best_MS + 1):
        #                     lines.append(stat_m[i])
        #                     labels.append("ms %d" % i)
        #                 util.plot(lines, labels, "Max juncture reached", pref="ms%d" %(ep // 1000))

    def report_stats(self, pref):
        super().report_stats(pref)
        
        #self.q_plotter.play_animation(save=True, pref="QLanes")
        #self.qx_plotter.play_animation(show=True, save=True, pref="QMaxLanes_%s" % pref)
        #self.c_plotter.play_animation(save=True, pref="CLanes")

        #         S_dqv = np.array(self.stat_debug_qv).T
        #         self.logger.debug("stat_debug_qv: \n%s", S_dqv.astype(np.int32))
        #         util.plot(S_dqv, self.stat_e_100, ["chosen", "ideal a", "ideal b"], 
        #                   "debug: q value", pref="dqv")
                   
        #         S_dn = np.array(self.stat_debug_n).T
        #         self.logger.debug("stat_debug_n: \n%s", (100*S_dn).astype(np.int32))
        #         util.plot(S_dn, self.stat_e_100, ["P(random action)"], 
        #                   "debug: n count", pref="dn")
     
        #         S_jmax = np.array(self.stat_juncture_maxQ).T
        #         self.logger.debug("stat_juncture_maxQ: \n%s", (100*S_jmax).astype(np.int32))
        #         labels = []
        #         for i in range(len(S_jmax)):
        #             labels.append("Jn %d" % i)
        #         util.plot(S_jmax, self.stat_e_100, labels, "Juncture max Q", pref="jmax")
        
        self.plotQ(28, (1, 4, 5), "speed, direction")
        self.plotQ(28, (1, 2, 3), "steering, accel") 
        #self.plotQ(28, (2, 3, 4, 5), "lanes") 

        #         S_qm = np.array(stat_qm).T
        #         logger.debug("stat_qm: \n%s", S_qm.astype(np.int32))
        #         util.heatmap(S_qm, (0, EPISODES, driver.num_junctures, 0),
        #                      "Total Q per juncture over epochs", pref="QM")
        #     #        
        #         #S_dq = S_qm[1:, :] - S_qm[:-1, :] # Q diff juncture to next juncture
        #         S_dq = S_qm[:, 1:] - S_qm[:, :-1] # Q diff epoch to next epoch
        #         logger.debug("stat_dq: \n%s", S_dq.shape)
        #         logger.debug("stat_dq: \n%s", S_dq.astype(np.int32))
        #         util.heatmap(S_dq, (0, EPISODES, driver.num_junctures, 0),
        #                      "Diff Q per juncture over epochs", pref="DQ")

        #util.plot([self.stat_dlm], self.stat_e_100, ["Avg ΔQ"], pref="delta")

    def plottable(self, X, axes, pick_max=False):
        X = np.max(X, axis=axes) if pick_max else np.sum(X, axis=axes)
        return X.reshape(X.shape[0], -1).T

    def plotQ(self, m, axes, t_suffix):
        Q = self.plottable(self.Q, axes)
        util.heatmap(Q, None, "Q total per juncture (%s)" % t_suffix,
                     pref="Q_%s" % t_suffix)
         
        C = self.plottable(self.C, axes)    
        util.heatmap(C, None, "Total updates per juncture (%s)" % t_suffix,
                     pref="C_%s" % t_suffix)

    def saveToFile(self, pref=""):
        M = self.num_junctures
        util.dump(self.Q.reshape(M, -1), "Q_%s_%s" % (M, pref))
        util.dump(self.N.reshape(M, -1), "N_%s_%s" % (M, pref))
