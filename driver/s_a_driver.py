'''
Created on Oct 30, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from . import Driver
import util

class SADriver(Driver):
    '''
    Base class for an agent that learns to drive a car along a track, 
    optimizing using a State-Action value-function rather than State alone.
    '''

    def __init__(self,
                 gamma,
                 explorate,
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
    
        util.pre_alg = "sa_%d" % (explorate)

        self.gamma = gamma  # weight given to predicted future
        self.explorate = explorate # Inclination to explore, e.g. 0, 10, 1000
        
        # Q is the learned value of a state/action
        self.Q = np.zeros((num_junctures,
                           num_lanes,
                           num_speeds,
                           num_directions,
                           num_steer_positions,
                           num_accel_positions))

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

    def _pick_action(self, S, run_best):
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

    def _max_at_state(self, S):
        return 0 if S is None else np.max(self.Q[S])
        
     
    def collect_stats(self, ep, num_episodes):
        super().collect_stats(ep, num_episodes)
        
        if util.checkpoint_reached(ep, num_episodes // 100):
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

        if util.checkpoint_reached(ep, num_episodes // 200):
            self.stat_e_200.append(ep)
    
            self.q_plotter.add_image(self._plottable(self.Q, (2, 3, 4, 5)))
            self.qx_plotter.add_image(self._plottable(self.Q, (2, 3, 4, 5), pick_max=True))
            self.c_plotter.add_image(self._plottable(self.C, (2, 3, 4, 5)))
        
#         if ep > 0 and ep % 10000 == 0:
#             logger.debug("Done %d episodes", ep)
#             #driver.plotQ(28, (1, 2, 3), "steering, accel") 
#             #driver.plotQ(30, (1, 4, 5), "speed, direction")
#             driver.plotQ(28, (2, 3, 4, 5), "lanes") 
#                  
#             lines = []
#             labels = []
#             for i in range(max(0, best_MS - 9), best_MS + 1):
#                 lines.append(stat_m[i])
#                 labels.append("ms %d" % i)
#             util.plot(lines, labels, "Max juncture reached", pref="ms%d" %(ep // 1000))

    def save_stats(self, pref=None):
        util.dump(np.asarray(self.stat_qm, dtype=np.float),
                  "dstats_qm", pref)
        util.dump(np.asarray(self.stat_cm, dtype=np.float),
                  "dstats_cm", pref)
        util.dump(np.asarray(self.stat_rm, dtype=np.float),
                  "dstats_rm", pref)
        util.dump(np.asarray(self.stat_juncture_maxQ, dtype=np.float),
                  "dstats_stat_juncture_maxQ", pref)
        
        A = np.asarray([
            self.stat_e_100,
            self.stat_dlm
            ], dtype=np.float)
        util.dump(A, "dstatsA", pref)

        C = np.asarray(self.stat_e_200, dtype=np.float)
        util.dump(C, "dstatsC", pref)
        
        self.q_plotter.save("q_plotter", pref)
        self.qx_plotter.save("qx_plotter", pref)
        self.c_plotter.save("c_plotter", pref)

    def load_stats(self, subdir, pref=None):
        self.stat_qm = list(util.load("dstats_qm", subdir, pref))
        self.stat_cm = list(util.load("dstats_cm", subdir, pref))
        self.stat_rm = list(util.load("dstats_rm", subdir, pref))
        self.dstats_stat_juncture_maxQ = list(util.load(
            "dstats_stat_juncture_maxQ", subdir, pref))
        
        A = util.load("dstatsA", subdir)
        self.stat_e_100 = list(A[0])
        self.stat_dlm = list(A[1])
        
        C = util.load("dstatsC", subdir)
        self.stat_e_200 = list(C)
        
        self.q_plotter.load("q_plotter", subdir, pref)
        self.qx_plotter.load("qx_plotter", subdir, pref)
        self.c_plotter.load("c_plotter", subdir, pref)
    
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
        
#         self._plotQ(28, (1, 4, 5), "speed, direction")
#         self._plotQ(28, (1, 2, 3), "steering, accel") 
        #self._plotQ(28, (2, 3, 4, 5), "lanes") 

#         S_qm = np.array(self.stat_qm).T
#         self.logger.debug("stat_qm: \n%s", S_qm.astype(np.int32))
#         util.heatmap(S_qm, (0, self.stat_e_100[-1], self.num_junctures, 0),
#                      "Total Q per juncture over epochs", pref="QM")
# 
#         S_cm = np.array(self.stat_cm).T
#         self.logger.debug("stat_cm: \n%s", S_cm.astype(np.int32))
#         util.heatmap(S_cm, (0, self.stat_e_100[-1], self.num_junctures, 0),
#                      "Exploration per juncture over epochs", pref="CM")
#       
#         S_rm = np.array(self.stat_rm).T
#         self.logger.debug("stat_rm: \n%s", (100*S_rm).astype(np.int32))
#         util.heatmap(S_rm, (0, self.stat_e_100[-1], self.num_junctures, 0),
#                      "Avg reward at juncture over epochs", cmap='winter',
#                      pref="RM")
        #        
        #         S_rm = S_rm[0:10]
        #         labels = []
        #         for i in range(len(S_rm)):
        #             labels.append("ms %d" % i)
        #         util.plot(S_rm, self.stat_e_100, labels, "Juncture reward", pref="rm")
        
        #
        #         #S_dq = S_qm[1:, :] - S_qm[:-1, :] # Q diff juncture to next juncture
        #         S_dq = S_qm[:, 1:] - S_qm[:, :-1] # Q diff epoch to next epoch
        #         logger.debug("stat_dq: \n%s", S_dq.shape)
        #         logger.debug("stat_dq: \n%s", S_dq.astype(np.int32))
        #         util.heatmap(S_dq, (0, EPISODES, driver.num_junctures, 0),
        #                      "Diff Q per juncture over epochs", pref="DQ")

        #util.plot([self.stat_dlm], self.stat_e_100, ["Avg ΔQ"], pref="delta")

    def _plottable(self, X, axes, pick_max=False):
        X = np.max(X, axis=axes) if pick_max else np.sum(X, axis=axes)
        return X.reshape(X.shape[0], -1).T

    def _plotQ(self, m, axes, t_suffix):
        Q = self._plottable(self.Q, axes)
        util.heatmap(Q, None, "Q total per juncture (%s)" % t_suffix,
                     pref="Q_%s" % t_suffix)
         
#         C = self._plottable(self.C, axes)    
#         util.heatmap(C, None, "Total updates per juncture (%s)" % t_suffix,
#                      pref="C_%s" % t_suffix)

    def save_model(self, pref=""):
        util.dump(self.Q, "Q")
        util.dump(self.N, "N")
        util.dump(self.C, "C")

    def load_model(self, load_subdir):
        self.logger.debug("Loading N, Q, C from: %s", load_subdir)
        self.Q = util.load("Q", load_subdir)
        self.N = util.load("N", load_subdir)
        self.C = util.load("C", load_subdir)
