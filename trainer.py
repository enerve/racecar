'''
Created on Sep 14, 2018

@author: enerve
'''

import logging
import numpy as np
import random
import time
import util

class Trainer:

    def __init__(self, driver, track, car):
        self.driver = driver
        self.track = track
        self.car = car
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
    def train(self, num_episodes, seed=10, pref='', save_to_file=True):
    
        random.seed(seed)
        
        # track counts of last juncture reached
        smooth = num_episodes // 100
        count_m = np.zeros((smooth, self.driver.num_junctures + 1), dtype=np.int32)
        Eye = np.eye(self.driver.num_junctures + 1)
        # track #steps/time-taken by bestpath, as iterations progress
        self.stat_bestpath_times = []
        # track recent average of rewards collected, as iterations progress
        self.stat_recent_total_R = []
        self.stat_bestpath_R = []
        self.stat_bestpath_juncture = []
        self.stat_e_bp = []
        
        self.stat_m = []#[] for _ in range(driver.num_junctures + 1)]
        self.stat_e_1000 = []
            
        self.logger.debug("Starting")
        start_time = time.clock()
        
        best_R = -10000
        best_ep = -1
        best_finished = False
        recent_total_R = 0
        
        num_restarts = 0
        for ep in range(num_episodes):
            total_R, environment = self.driver.run_episode(self.track, self.car)
    
            count_m[ep % smooth] = Eye[environment.curr_juncture]
            
            if environment.curr_juncture >= 10:
                if total_R > best_R:
                    self.logger.debug("Ep %d  Juncture %d reached with R=%d T=%d",
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
        #            elif ep - best_ep > 3000:                
        #                logger.debug("Restarting exploration (ep %d)", ep)
        #                driver.restart_exploration(scale_explorate=1.5)
        #                 num_restarts += 1
        #                 
        #                 if ep > num_episodes:
        #                     break
        #                best_R = -10000
        #                best_ep = -1
        #                best_finished = False
                    
            self.driver.collect_stats(ep, num_episodes)
                    
    
            if util.checkpoint_reached(ep, num_episodes // 200):
                self.stat_e_1000.append(ep)
                self.stat_m.append(count_m.sum(axis=0))
                #for sm, c in zip(stat_m, count_m.sum(axis=0)):
                #    sm.append(c)
                    #stat_ep[i].append(ep)
                #count_m = np.zeros((driver.num_junctures + 1), dtype=np.int32)
    
                
            len_bp_split = (num_episodes // 100)
            recent_total_R += (total_R - recent_total_R) * 10 / len_bp_split
            if ep > 0 and ep % len_bp_split == len_bp_split - 1:
                bestpath_R, bestpath_env = self.driver.run_episode(self.track, 
                                                                   self.car, 
                                                                   run_best=True)
                self.stat_bestpath_times.append(bestpath_env.total_time_taken() 
                                           if bestpath_env.has_reached_finish() 
                                           else 500)
                self.stat_bestpath_R.append(bestpath_R)
                self.stat_e_bp.append(ep)
                self.stat_recent_total_R.append(recent_total_R)
                self.stat_bestpath_juncture.append(bestpath_env.curr_juncture)
                
    
            if util.checkpoint_reached(ep, 1000):
                self.logger.debug("Ep %d ", ep)
                
    
        self.logger.debug("Completed training in %d seconds", time.clock() - start_time)
    
        if save_to_file:
            # save learned values to file
            self.driver.save_model(pref=pref)
            
            # save stats to file
            self.save_stats(pref=pref)        
    
        return (self.stat_bestpath_times, self.stat_e_bp,
                self.stat_bestpath_R, self.stat_bestpath_juncture)
    
    
    def save_stats(self, pref=None):
        self.driver.save_stats(pref=pref)
        
        A = np.asarray([
            self.stat_bestpath_times,
            self.stat_recent_total_R,
            self.stat_e_bp,
            self.stat_bestpath_R,
            ], dtype=np.float)
        util.dump(A, "statsA", pref)
    
        B = np.asarray(self.stat_m, dtype=np.float)
        util.dump(B, "statsB", pref)
    
        C = np.asarray(self.stat_e_1000, dtype=np.float)
        util.dump(C, "statsC", pref)
        
    def load_stats(self, subdir, pref=None):
        self.driver.load_stats(subdir, pref)
    
        self.logger.debug("Loading stats...")
        
        A = util.load("statsA", subdir, pref)
        self.stat_bestpath_times = A[0]
        self.stat_recent_total_R = A[1]
        self.stat_e_bp = A[2]
        self.stat_bestpath_R = A[3]
        
        B = util.load("statsB", subdir, pref)
        self.stat_m = B
        
        C = util.load("statsC", subdir, pref)
        self.stat_e_1000 = C
    
    def report_stats(self, pref=None):
        self.driver.report_stats(pref=pref)
        
        util.plot([[10*x for x in self.stat_bestpath_times],
                   self.stat_recent_total_R, self.stat_bestpath_R],
                  self.stat_e_bp,
                  ["Finish time by best-action path", "Recent avg total reward",
                   "Reward for best-action path"],
                  title="Performance over time",
                  pref="bpt")
        
        # Max juncture reached
    #     for i in range(len(stat_m)):
    #         lines.append(stat_m[i])
        S_m = np.array(self.stat_m).T
        labels = ["ms %d" % i for i in range(len(S_m))]
        util.plot(S_m, self.stat_e_1000, labels, "Max juncture reached", pref="ms")
    
    def play_best(self, should_play_movie=True, pref=""):
        #logger.debug("Playing best path")
        environment = self.best_environment()
        environment.report_history()
        environment.play_movie(show=should_play_movie, pref="bestmovie_%s" % pref)
        return environment
    
    def show_best(self, show=True, pref=""):
        #logger.debug("Playing best path")
        environment = self.best_environment()
        environment.report_history()
        environment.show_path(show=show, pref="bestpath_%s" % pref)
        return environment
    
    def best_environment(self):
        total_R, environment = self.driver.run_episode(self.track, self.car, 
                                                       run_best=True)
        return environment
