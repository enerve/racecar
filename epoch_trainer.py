'''
Created on Nov 6, 2018

@author: enerve
'''

import logging
import numpy as np
import time
import util

class EpochTrainer:
    ''' A class that helps train the RL agent in stages, collecting episode
        history for an epoch and then training on that data.
    '''

    def __init__(self, driver, track, car):
        self.driver = driver
        self.track = track
        self.car = car
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
    def train(self, num_epochs, num_episodes_per_epoch, pref='', save_to_file=True):
        
        total_episodes = num_epochs * num_episodes_per_epoch

        # track counts of last juncture reached
        smooth = total_episodes // 100
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

        ep = 0
        for epoch in range(num_epochs):
            # In each epoch, we first collect experience, then (re)train FA
            self.logger.debug("====== Epoch %d =====", epoch)
            
            history = [] # history of steps in episodes, each containing
                         # state, action, reward and next state
            
            for ep_ in range(num_episodes_per_epoch):
                total_R, environment, ep_hist = self.driver.run_episode(
                    self.track, self.car)
                history.extend(ep_hist)
        
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
                        
                # TODO: fix
                self.driver.collect_stats(ep, total_episodes)
                        
        
                if util.checkpoint_reached(ep, total_episodes // 200):
                    self.stat_e_1000.append(ep)
                    self.stat_m.append(count_m.sum(axis=0))
                    #for sm, c in zip(stat_m, count_m.sum(axis=0)):
                    #    sm.append(c)
                        #stat_ep[i].append(ep)
                    #count_m = np.zeros((driver.num_junctures + 1), dtype=np.int32)
        
                    
                len_bp_split = (total_episodes // 100)
                recent_total_R += (total_R - recent_total_R) * 10 / len_bp_split
                if ep > 0 and ep % len_bp_split == len_bp_split - 1:
                    bestpath_R, bestpath_env, _ = self.driver.run_episode(self.track, 
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
                    
                ep += 1
            
            
            self.driver.update_fa()
            
#             if self.student_driver:
#                 self.student_driver.observe_history(history)
#                 self.student_driver.update_fa()
            
            #    self.driver.restart_exploration()
                    
    
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
        
        # TODO: Save and load #episodes done in total
        
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
        
#         util.plot([[x for x in self.stat_bestpath_times],
#                    self.stat_recent_total_R, self.stat_bestpath_R],
#                   self.stat_e_bp,
#                   ["Finish time by best-action path", "Recent avg total reward",
#                    "Reward for best-action path"],
#                   title="Performance over time",
#                   pref="bpt")
        
        # Max juncture reached
    #     for i in range(len(stat_m)):
    #         lines.append(stat_m[i])
#         S_m = np.array(self.stat_m).T
#         labels = ["ms %d" % i for i in range(len(S_m))]
#         util.plot(S_m, self.stat_e_1000, labels, "Max juncture reached", pref="ms")
    
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
