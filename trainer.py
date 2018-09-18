'''
Created on Sep 14, 2018

@author: enerve
'''

import logging
import numpy as np
import random

import util

def train(driver, track, car, num_episodes, seed=10, pref=''):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    random.seed(seed)
    
    # track counts of last juncture reached
    smooth = num_episodes // 100
    count_m = np.zeros((smooth, driver.num_junctures + 1), dtype=np.int32)
    Eye = np.eye(driver.num_junctures + 1)
    # track #steps/time-taken by bestpath, as iterations progress
    stat_bestpath_times = []
    # track recent average of rewards collected, as iterations progress
    stat_recent_total_R = []
    # track max Q per juncture, as iterations progress
    stat_juncture_maxQ = []
    stat_e_bp = []
    
    stat_m = [[] for _ in range(driver.num_junctures + 1)]
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
    best_ep = -1
    best_finished = False
    recent_total_R = 0
    
    NUM_RESTARTS = 6
    for ep in range(num_episodes):
        total_R, environment = driver.run_episode(track, car)

            
        count_m[ep % smooth] = Eye[environment.curr_juncture]
        
        if environment.curr_juncture >= 10:
            if total_R > best_R:
                logger.debug("Ep %d  Juncture %d reached with R=%d T=%d",
                                  ep, environment.curr_juncture, total_R,
                                  environment.total_time_taken())
                best_R = total_R
                #best_juncture = environment.curr_juncture
                best_ep = ep
                best_finished = environment.has_reached_finish()
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
            elif best_finished and ep - best_ep > 3000:                
                logger.debug("Restarting exploration (ep %d)", ep)
                driver.restart_exploration(scale_explorate=1.5)
                best_R = -10000
                best_ep = -1
                best_finished = False
                

        if ep > 0 and ep % (num_episodes // 1000) == 0:
            stat_e_1000.append(ep)
            for i, c in enumerate(count_m.sum(axis=0)):
                stat_m[i].append(c)
#                     stat_ep[i].append(ep)
            #count_m = np.zeros((driver.num_junctures + 1), dtype=np.int32)

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
            
        len_bp_split = (num_episodes // 100)
        recent_total_R += (total_R - recent_total_R) * 10 / len_bp_split
        if ep > 0 and ep % len_bp_split == len_bp_split - 1:
            bestpath_env = best_environment(driver, track, car)
            if bestpath_env.has_reached_finish():
                stat_bestpath_times.append(bestpath_env.total_time_taken())
            else:
                stat_bestpath_times.append(500)
            stat_e_bp.append(ep)
            stat_recent_total_R.append(recent_total_R)
            stat_juncture_maxQ.append(np.max(driver.Q, axis=(1,2,3,4,5)))
            
        if ep > 0 and ep % (num_episodes // 200) == 0:
            stat_e_200.append(ep)
    
            q_plotter.add_image(driver.Q_to_plot((2, 3, 4, 5)))
            qx_plotter.add_image(driver.Q_to_plot((2, 3, 4, 5), pick_max=True))
            c_plotter.add_image(driver.C_to_plot((2, 3, 4, 5)))
        
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
    qx_plotter.play_animation(show=True, save=True, pref="QMaxLanes_%s" % pref)
    #c_plotter.play_animation(save=True, pref="CLanes")
    
#     S_qm = np.array(stat_qm).T
#     logger.debug("stat_qm: \n%s", S_qm.astype(np.int32))
#     util.heatmap(S_qm, (0, EPISODES, driver.num_junctures, 0),
#                  "Total Q per juncture over epochs", pref="QM")
#        
#     #S_dq = S_qm[1:, :] - S_qm[:-1, :] # Q diff juncture to next juncture
#     S_dq = S_qm[:, 1:] - S_qm[:, :-1] # Q diff epoch to next epoch
#     logger.debug("stat_dq: \n%s", S_dq.shape)
#     logger.debug("stat_dq: \n%s", S_dq.astype(np.int32))
#     util.heatmap(S_dq, (0, EPISODES, driver.num_junctures, 0),
#                  "Diff Q per juncture over epochs", pref="DQ")
#  
#     S_cm = np.array(stat_cm).T
#     logger.debug("stat_cm: \n%s", S_cm.astype(np.int32))
#     util.heatmap(S_cm, (0, EPISODES, driver.num_junctures, 0),
#                  "Exploration per juncture over epochs", pref="CM")
# 
#     S_rm = np.array(stat_rm).T
#     logger.debug("stat_rm: \n%s", (100*S_rm).astype(np.int32))
#     util.heatmap(S_rm, (0, EPISODES, driver.num_junctures, 0),
#                  "Avg reward at juncture over epochs", cmap='winter',
#                  pref="RM")
 
#     S_rm = S_rm[0:10]
#     labels = []
#     for i in range(len(S_rm)):
#         labels.append("ms %d" % i)
#     util.plot(S_rm, stat_e_100, labels, "Juncture reward", pref="rm")

    util.plot([[10*x for x in stat_bestpath_times], stat_recent_total_R], stat_e_bp,
              ["Finish time by best-action path", "Recent avg total reward"],
              title="Performance over time",
              pref="bpt")
    
    S_jmax = np.array(stat_juncture_maxQ).T
    logger.debug("stat_juncture_maxQ: \n%s", (100*S_jmax).astype(np.int32))
    labels = []
    for i in range(len(S_jmax)):
        labels.append("Jn %d" % i)
    util.plot(S_jmax, stat_e_bp, labels, "Juncture max Q", pref="jmax")
    
    # Max juncture reached
    lines = []
    labels = []
    for i in range(len(stat_m)):
        lines.append(stat_m[i])
        labels.append("ms %d" % i)            
    util.plot(lines, stat_e_1000, labels, "Max juncture reached", pref="ms")

    return (stat_bestpath_times, stat_e_bp)

def play_best(driver, track, car, should_play_movie=True, pref=""):
    #logger.debug("Playing best path")
    environment = best_environment(driver, track, car)
    environment.report_history()
    environment.play_movie(show=should_play_movie, pref="bestmovie_%s" % pref)
    return environment

def best_environment(driver, track, car):
    total_R, environment = driver.run_episode(track, car, run_best=True)
    return environment

def drive_manual(environment, manual_actions):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    for A in manual_actions:
        R, S_ = environment.step(A)
        logger.debug(" A:%s  R:%d    S:%s" % (A, R, S_,))
        if S_ is None:
            break

    environment.report_history()
    environment.play_movie(save=False, pref="bestmovie")

