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

    def __init__(self, driver, track, car, student=None):
        self.driver = driver
        self.student = student
        self.track = track
        self.car = car
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_driver_alg = self.driver.prefix()
        self.logger.debug("Driver: %s", util.pre_driver_alg)
        if self.student:
            util.pre_student_alg = self.student.prefix()
            self.logger.debug("Student: %s", util.pre_student_alg)

        # track #steps/time-taken by bestpath, as iterations progress
        self.stat_bestpath_times = []
        # track recent average of rewards collected, as iterations progress
        self.stat_recent_total_R = []
        self.stat_bestpath_R = []
        self.stat_bestpath_juncture = []
        self.stat_e_bp = []

        # track mimicFA performance
        self.stat_mimic_bestpath_times = []
        # track mimicFA recent average of rewards collected, as iterations progress
        self.stat_mimic_bestpath_R = []

        # track student peformance
        self.stat_student_bestpath_times = []
        # track recent average of rewards collected, as iterations progress
        self.stat_student_bestpath_R = []
        
        self.stat_m = []#[] for _ in range(driver.num_junctures + 1)]
        self.stat_e_1000 = []
        
        self.ep = 0

    def train(self, num_epochs, num_episodes_per_epoch, num_explorations = 1,
              store_training_data = False):
        
        total_episodes = num_epochs * num_episodes_per_epoch * num_explorations

        # track counts of last juncture reached
        smooth = total_episodes // 100
        count_m = np.zeros((smooth, self.driver.num_junctures + 1), dtype=np.int32)
        Eye = np.eye(self.driver.num_junctures + 1)
            
        self.logger.debug("Starting for %d expls x %d epochs x %d episodes",
                          num_explorations, num_epochs, num_episodes_per_epoch)
        start_time = time.clock()
        
        recent_total_R = 0

        ep = ep_s = self.ep
        for expl in range(num_explorations):
            for epoch in range(num_epochs):
                # In each epoch, we first collect experience, then (re)train FA
                self.logger.debug("====== Expl %d epoch %d =====", expl, epoch)
                
                history = [] # history of episodes, each episode a list of tuples of
                             # state, action, reward

                best_R = -10000
                
                for ep_ in range(num_episodes_per_epoch):
                    total_R, environment, ep_steps = self.driver.run_episode(
                        self.track, self.car)
                    history.append(ep_steps)
            
                    count_m[ep % smooth] = Eye[environment.curr_juncture]
                    
                    if environment.curr_juncture >= 6:
                        if total_R > best_R:
                            self.logger.debug("Ep %d  Juncture %d reached with tR=%d T=%d",
                                              ep, environment.curr_juncture, total_R,
                                              environment.total_time_taken())
                            best_R = total_R
                            
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
                    if (ep + 1) % len_bp_split == 0:
                        bestpath_R, bestpath_env, _ = self.driver.run_best_episode(
                            self.track, self.car)
                        self.stat_bestpath_times.append(bestpath_env.total_time_taken() 
                                                   if bestpath_env.has_reached_finish() 
                                                   else 500)
                        self.stat_bestpath_R.append(bestpath_R)
                        self.stat_e_bp.append(ep)
                        self.stat_recent_total_R.append(recent_total_R)
                        self.stat_bestpath_juncture.append(bestpath_env.curr_juncture)
                        if self.driver.mimic_fa:
                            bestpath_R, bestpath_env, _ = self.driver.run_best_episode(
                                self.track, self.car, use_mimic=True)
                            self.stat_mimic_bestpath_times.append(bestpath_env.total_time_taken() 
                                                       if bestpath_env.has_reached_finish() 
                                                       else 500)
                            self.stat_mimic_bestpath_R.append(bestpath_R)

                        if self.student:
                            bestpath_R, bestpath_env, _ = self.student.run_best_episode(
                                self.track, self.car)
                            self.stat_student_bestpath_times.append(
                                bestpath_env.total_time_taken() if 
                                bestpath_env.has_reached_finish() else 500)
                            self.stat_student_bestpath_R.append(bestpath_R)
                        
            
                    if util.checkpoint_reached(ep, 1000):
                        self.logger.debug("Ep %d ", ep)
                        
                    ep += 1
                
                
                self.driver.update_fa()
                
                if self.student:
                    # Train the student driver with the history of episodes
                    for ep_steps in history:
                        self.student.observe_episode(ep_steps)
                        self.student.collect_stats(ep_s, total_episodes)
                        ep_s += 1
                    self.student.update_fa()
            
            self.driver.restart_exploration(1) #(1.5)

        self.logger.debug("Completed training in %d seconds", time.clock() - start_time)
    
        self.ep = ep
        
        return (self.stat_bestpath_times, self.stat_e_bp,
                self.stat_bestpath_R, self.stat_bestpath_juncture)
    
    def load_from_file(self, subdir):
        self.driver.load_model(subdir)
        #self.load_stats(subdir)

    def save_to_file(self, pref=''):
        # save learned values to file
        self.driver.save_model(pref=pref)
        
        # save stats to file
        self.save_stats(pref=pref)        
    
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
    
    def report_stats(self, pref=""):
        self.driver.report_stats(pref="g_" + pref)
        
        if self.student:
            self.student.report_stats(pref="s_" + pref)
        util.plot([[x for x in self.stat_bestpath_times],
                   self.stat_recent_total_R, self.stat_bestpath_R],
                  self.stat_e_bp,
                  ["Finish time by best-action path", "Recent avg total reward",
                   "Reward for best-action path"],
                  title="Performance over time",
                  pref="bpt")
        if self.driver.mimic_fa:
            util.plot([[x for x in self.stat_mimic_bestpath_times],
                       self.stat_recent_total_R, self.stat_mimic_bestpath_R],
                      self.stat_e_bp,
                      ["Finish time by best-action path", "Recent avg total reward",
                       "Reward for best-action path"],
                      title="Mimic Performance over time",
                      pref="m_bpt")
        if self.student:
            util.plot([self.stat_student_bestpath_times,
                       self.stat_recent_total_R, self.stat_student_bestpath_R],
                      self.stat_e_bp,
                      ["Finish time by best-action path", "Recent avg total reward",
                       "Reward for best-action path"],
                      title="Student Performance over time",
                      pref="s_bpt")
        
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
        total_R, environment, _ = self.driver.run_best_episode(self.track, self.car)
        return environment
