'''
Created on Sep 6, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from environment import Environment
from car import Car
from track import CircleTrack
import cmd_line
import log
import util

class Driver(object):
    '''
    An agent that controls the race car
    '''

    NUM_JUNCTURES = 28
    NUM_LANES = 5
    MAX_SPEED = 3
    NUM_DIRECTIONS = 20
    
    NUM_STEER_POSITIONS = 3
    NUM_ACCEL_POSITIONS = 3

    def __init__(self, alpha, gamma, explorate, load_filename=None):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_alg = "qlearn"
        
        self.alpha = alpha  # learning rate for updating Q
        self.gamma = gamma  # weight given to predicted future
        self.explorate = explorate # Inclination to explore, e.g. 0, 10, 1000
        
        # Q is the learned value of a state/action
        if load_filename is not None:
            stored_Q = util.load(load_filename)
            self.Q = stored_Q.reshape((Driver.NUM_JUNCTURES,
                                       Driver.NUM_LANES,
                                       Driver.MAX_SPEED,
                                       Driver.NUM_DIRECTIONS,
                                       Driver.NUM_STEER_POSITIONS,
                                       Driver.NUM_ACCEL_POSITIONS))
        else:
            self.Q = np.zeros((Driver.NUM_JUNCTURES,
                               Driver.NUM_LANES,
                               Driver.MAX_SPEED,
                               Driver.NUM_DIRECTIONS,
                               Driver.NUM_STEER_POSITIONS,
                               Driver.NUM_ACCEL_POSITIONS))
                           
        # C is the count of visits to state/action
        self.C = np.zeros((Driver.NUM_JUNCTURES,
                           Driver.NUM_LANES,
                           Driver.MAX_SPEED,
                           Driver.NUM_DIRECTIONS,
                           Driver.NUM_STEER_POSITIONS,
                           Driver.NUM_ACCEL_POSITIONS), dtype=np.int32)
        # C is the count of visits to state
        self.N = np.zeros((Driver.NUM_JUNCTURES,
                           Driver.NUM_LANES,
                           Driver.MAX_SPEED,
                           Driver.NUM_DIRECTIONS), dtype=np.int32)
        # Rs is the average reward at juncture (for statistics)
        self.Rs = np.zeros((Driver.NUM_JUNCTURES), dtype=np.float)

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
                Driver.NUM_STEER_POSITIONS * Driver.NUM_ACCEL_POSITIONS)
            steer = r % Driver.NUM_STEER_POSITIONS
            accel = r // Driver.NUM_STEER_POSITIONS
        
        return (steer, accel)
    
    def max_at_state(self, S):
        if S is None:
            return 0
        # TODO: Add some kind of nearby-approximation?
        return np.max(self.Q[S])
        
    def q_index(self, state, action):
        m, l, v, d = state
        s, a = action
        return (m, l, v, d, s, a)
    
    def Q_to_plot(self, m, axes, pick_max=False):
        M = m
        Q = self.Q[0:M, :, :, :, :, :]
        if pick_max:
            Q = np.max(Q, axis=axes)
        else:
            Q = np.mean(Q, axis=axes)
        Q = Q.reshape(M, -1).T
        #self.logger.debug("%s", Q.astype(np.int32))
        return Q

    def C_to_plot(self, m, axes):
        M = m
        C = self.C[0:M, :, :, :, :, :]
        C = np.sum( C, axis=axes)
        C = C.reshape(M, -1).T
        #self.logger.debug("%s", C)
        return C

    def plotQ(self, m, axes, t_suffix):
        M = m
        Q = self.Q_to_plot(m, axes)
        util.heatmap(Q, None, "Q total per juncture %s" % t_suffix, pref="Q")
        
        C = self.C_to_plot(m, axes)    
        util.heatmap(C, None, "Total updates per juncture %s" % t_suffix, pref="C")

    def dumpQ(self, pref=""):
        M = Driver.NUM_JUNCTURES
        util.dump(self.Q.reshape(M, -1), "Q_%s_%s" % (M, pref))
        #util.dump(self.C.reshape(M, -1), "C_%s" % M)
        #util.dump(self.N.reshape(M, -1), "N_%s" % M)

    def run_episode(self, track, car, run_best=False):
        environment = Environment(track,
                                  car,
                                  Driver.NUM_JUNCTURES,
                                  should_record=run_best)
        S = environment.state_encoding() # 0 2 0 0
        total_R = 0
        
        while S is not None:
            A = self.pick_action(S, run_best)
            
            I = self.q_index(S, A)

            R, S_ = environment.step(A)

            target = R + self.gamma * self.max_at_state(S_)
            self.Q[I] += self.alpha * (target - self.Q[I])
            self.C[I] += 1
            self.N[S] += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            
            S = S_
            total_R += R
            
        return total_R, environment
            

# ----------------------------------------------------------------------------
            
def train(driver, track, car, num_episodes, seed=10, pref=''):
    
    random.seed(seed)
    
    # track counts of last juncture reached
    smooth = num_episodes // 100
    count_m = np.zeros((smooth, Driver.NUM_JUNCTURES + 1), dtype=np.int32)
    Eye = np.eye(Driver.NUM_JUNCTURES + 1)
    # track #steps/time-taken by bestpath, as iterations progress
    stat_bestpath_times = []
    stat_e_bp = []
    
    stat_m = [[] for _ in range(Driver.NUM_JUNCTURES + 1)]
    stat_qm = []
    stat_cm = []
    stat_rm = []
    stat_debug_qv = []
    stat_debug_n = []
    q_plotter = util.Plotter("Q value at junctures along lanes")
    qx_plotter = util.Plotter("Max Q value at junctures along lanes")
    c_plotter = util.Plotter("C value at junctures along lanes")
    stat_e_100 = []
    stat_e_200 = []
    stat_e_1000 = []
        
    logger.debug("Starting")
    best_R = -10000
    best_Jc = 9
    
    for ep in range(num_episodes):
        R, environment = driver.run_episode(track, car)

            
        count_m[ep % smooth] = Eye[environment.curr_juncture]
        
        if environment.curr_juncture >= 10:
            if R > best_R:
                logger.debug("Ep %d  Juncture %d reached with R=%d T=%d",
                                  ep, environment.curr_juncture, R,
                                  environment.total_time_taken())
                best_R = R
                best_Jc = environment.curr_juncture
#                     environment.report_history()
#                     environment.play_movie()

#             if environment.curr_juncture >= 10:
#                 ah = environment.car.action_history
#                 pattern = [2]
#                 still_matching = True
#                 for i, p in enumerate(pattern):
#                     #a = ah[i][0]  # steer
#                     a = ah[i][1]  # accel
#                     if p != a:
#                         still_matching = False
#                         break;
#                 if still_matching:
#                     logger.debug("Ep %d  juncture %d reached with R=%d",
#                                        ep, environment.curr_juncture, R)
#                     environment.report_history()
#                     environment.play_movie()

        if ep > 0 and ep % (num_episodes // 1000) == 0:
            stat_e_1000.append(ep)
            for i, c in enumerate(count_m.sum(axis=0)):
                stat_m[i].append(c)
#                     stat_ep[i].append(ep)
            #count_m = np.zeros((Driver.NUM_JUNCTURES + 1), dtype=np.int32)

        if ep > 0 and ep % (num_episodes // 100) == 0:
            stat_e_100.append(ep)

            stat_qm.append(driver.Q.sum(axis=(1,2,3,4,5)))
            stat_cm.append((driver.C/(driver.C+1)).sum(axis=(1,2,3,4,5)))
            stat_rm.append(driver.Rs.copy())

            av = driver.Q[3, 1, 2, 2, 1, 2]  # what is chosen best in the end
            bv = driver.Q[3, 1, 2, 2, 0, 1]  # what i hoped for
            cv = driver.Q[3, 1, 2, 2, 0, 2]  #     or this
            stat_debug_qv.append([av, bv, cv])

            an = driver.N[3, 1, 2, 2]  # what is chosen best in the end
#             bn = driver.C[3, 1, 2, 2, 0, 1]  # what i hoped for
#             cn = driver.C[3, 1, 2, 2, 0, 2]  #     or this
            an = driver.explorate / (driver.explorate + an)
            stat_debug_n.append([an])
            
        if ep > 0 and ep % (num_episodes // 100) == 99:
            bestpath_env = best_environment(driver, track, car)
            if bestpath_env.has_reached_finish():
                stat_bestpath_times.append(bestpath_env.total_time_taken())
                stat_e_bp.append(ep)
            
        if ep > 0 and ep % (num_episodes // 200) == 0:
            stat_e_200.append(ep)
    
            q_plotter.add_image(driver.Q_to_plot(28, (2, 3, 4, 5)))
            qx_plotter.add_image(driver.Q_to_plot(28, (2, 3, 4, 5), pick_max=True))
            c_plotter.add_image(driver.C_to_plot(28, (2, 3, 4, 5)))
        
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

        if ep > 0 and ep % 10000 == 0:
            logger.debug("Ep %d ", ep)

    # save learned values to file
    driver.dumpQ(pref=pref)

#     S_dqv = np.array(stat_debug_qv).T
#     logger.debug("stat_debug_qv: \n%s", S_dqv.astype(np.int32))
#     util.plot(S_dqv, stat_e_100, ["chosen", "ideal a", "ideal b"], 
#               "debug: q value", pref="dqv")
#       
#     S_dn = np.array(stat_debug_n).T
#     logger.debug("stat_debug_n: \n%s", (100*S_dn).astype(np.int32))
#     util.plot(S_dn, stat_e_100, ["P(random action)"], 
#               "debug: n count", pref="dn")
# 
#     driver.plotQ(28, (1, 4, 5), "speed, direction")
#     driver.plotQ(28, (1, 2, 3), "steering, accel") 
#     driver.plotQ(28, (2, 3, 4, 5), "lanes") 
    
    #q_plotter.play_animation(save=True, pref="QLanes")
    qx_plotter.play_animation(show=False, save=True, pref="QMaxLanes_%s" % pref)
    #c_plotter.play_animation(save=True, pref="CLanes")
    
#     S_qm = np.array(stat_qm).T
#     logger.debug("stat_qm: \n%s", S_qm.astype(np.int32))
#     util.heatmap(S_qm, (0, EPISODES, Driver.NUM_JUNCTURES, 0),
#                  "Total Q per juncture over epochs", pref="QM")
#        
#     #S_dq = S_qm[1:, :] - S_qm[:-1, :] # Q diff juncture to next juncture
#     S_dq = S_qm[:, 1:] - S_qm[:, :-1] # Q diff epoch to next epoch
#     logger.debug("stat_dq: \n%s", S_dq.shape)
#     logger.debug("stat_dq: \n%s", S_dq.astype(np.int32))
#     util.heatmap(S_dq, (0, EPISODES, Driver.NUM_JUNCTURES, 0),
#                  "Diff Q per juncture over epochs", pref="DQ")
#  
#     S_cm = np.array(stat_cm).T
#     logger.debug("stat_cm: \n%s", S_cm.astype(np.int32))
#     util.heatmap(S_cm, (0, EPISODES, Driver.NUM_JUNCTURES, 0),
#                  "Exploration per juncture over epochs", pref="CM")
# 
#     S_rm = np.array(stat_rm).T
#     logger.debug("stat_rm: \n%s", (100*S_rm).astype(np.int32))
#     util.heatmap(S_rm, (0, EPISODES, Driver.NUM_JUNCTURES, 0),
#                  "Avg reward at juncture over epochs", cmap='winter',
#                  pref="RM")
 
#     S_rm = S_rm[0:10]
#     labels = []
#     for i in range(len(S_rm)):
#         labels.append("ms %d" % i)
#     util.plot(S_rm, stat_e_100, labels, "Juncture reward", pref="rm")

#     lines = []
#     labels = []
#     for i in range(len(stat_m)):
#         lines.append(stat_m[i])
#         labels.append("ms %d" % i)            
#     util.plot(lines, stat_e_1000, labels, "Max juncture reached", pref="ms")

    return (stat_bestpath_times, stat_e_bp)

def play_best(driver, track, car, should_play_movie=True, pref=""):
    #logger.debug("Playing best path")
    environment = best_environment(driver, track, car)
    environment.report_history()
    environment.play_movie(show=should_play_movie, pref="bestmovie_%s" % pref)
    return environment

def best_environment(driver, track, car):
    R, environment = driver.run_episode(track, car, run_best=True)
    return environment

MY_IDEAL_A = [
            (0, 2),
            (1, 2),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ]


def drive_manual(track, car):
    environment = Environment(track, car, Driver.NUM_JUNCTURES,
                              should_record=True)

    actions = MY_IDEAL_A
    for A in actions:
        R, S_ = environment.step(A)
        logger.debug(" A:%s  R:%d    S:%s" % (A, R, S_,))

    environment.report_history()
    environment.play_movie(save=False, pref="bestmovie")

            
if __name__ == '__main__':
    
    args = cmd_line.parse_args()

    util.prefix_init(args)
    util.pre_problem = 'RC'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar FS")
    logger.setLevel(logging.DEBUG)

    # --------------
    
    RADIUS = 98
    track = CircleTrack((0, 0), RADIUS, 20, Driver.NUM_JUNCTURES,
                        Environment.NUM_MILESTONES, Driver.NUM_LANES)
    NUM_SPEEDS = 3
    car = Car(Driver.NUM_DIRECTIONS, NUM_SPEEDS)
    
    
    #original_driver = Driver(alpha=1, gamma=1, explorate=2500)
    driver = Driver(alpha=1, gamma=1, explorate=2500)
    #driver = Driver(alpha=0.2, gamma=1, load_filename="RC_qlearn_652042_Q_28_.csv")
    train(driver, track, car, 200*1000)
    play_best(driver, track, car)
            
    # drive_manual()

    # --------- CV ---------
#     explorates = [10, 100, 1000, 10000]
#     stats_bp_times = []
#     stats_e_bp = []
#     stats_labels = []
#     num_episodes = 300 * 1000
#     for explorate in explorates:
#         logger.debug("--- Explorate=%d ---" % explorate)
#         for i in range(3):
#             seed = (100 + 53*i)
#             pref = "%d_%d" % (explorate, seed)
#             driver = Driver(alpha=1, gamma=1, explorate=explorate)
#             stat_bestpath_times, stat_e_bp = \
#                 train(driver, track, car, num_episodes, seed=seed, pref=pref)
#             stats_bp_times.append(stat_bestpath_times)
#             stats_e_bp.append(stat_e_bp)
#             stats_labels.append("N0=%d seed=%d" % (explorate, seed))
#             logger.debug("bestpath: %s", stat_bestpath_times)
#             logger.debug("stat_e: %s", stat_e_bp)
#             play_best(driver, track, car, should_play_movie=False,
#                       pref=pref)
#     util.plot_all(stats_bp_times, stats_e_bp, stats_labels,
#                   title="Time taken by best path as of epoch", pref="BestTimeTaken")
    
    

            