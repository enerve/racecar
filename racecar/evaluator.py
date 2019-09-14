'''
Created on 23 May 2019

@author: enerve
'''

import numpy as np

import logging
from really import util
from really.evaluator import Evaluator as Eval

class Evaluator(Eval):
    
    def __init__(self, episode_factory, test_agent):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.episode_factory = episode_factory
        self.test_agent = test_agent

        # track #steps/time-taken by bestpath, as iterations progress
        self.stat_bestpath_times = []
        # track recent average of rewards collected, as iterations progress
        self.stat_bestpath_R = []
        self.stat_bestpath_juncture = []
        self.stat_e_bp = []

        self.q_plotter = util.Plotter("Q value at junctures along lanes")
        self.c_plotter = util.Plotter("C value at junctures along lanes")

    def evaluate(self, ep): #TODO: rename to collect_stats??
        episode = self.episode_factory.new_episode([self.test_agent])
        episode.run()
        score = self.test_agent.G

        self.logger.debug("Tester score (G): %0.2f  and reached juncture %d" % 
                          (score, episode.curr_juncture))

        self.stat_bestpath_times.append(episode.total_time_taken()
                                        if episode.has_reached_finish()
                                        else 500)
        self.stat_bestpath_R.append(self.test_agent.G)
        self.stat_e_bp.append(ep)
        self.stat_bestpath_juncture.append(episode.curr_juncture)
        
#         self.q_plotter.add_image(self._plottable(self.Q, (2, 3, 4, 5)))
#         self.c_plotter.add_image(self._plottable(self.C, (2, 3, 4, 5)))


    def save_stats(self, pref=None):
        self.test_agent.save_stats(pref=pref)
        
        A = np.asarray([
            self.stat_bestpath_times,
            self.stat_e_bp,
            self.stat_bestpath_R,
            ], dtype=np.float)
        util.dump(A, "statsA", pref)
    
#         B = np.asarray(self.stat_m, dtype=np.float)
#         util.dump(B, "statsB", pref)
#     
#         C = np.asarray(self.stat_e_1000, dtype=np.float)
#         util.dump(C, "statsC", pref)
        
        # TODO: Save and load #episodes done in total
        
    def load_stats(self, subdir, pref=None):
        self.test_agent.load_stats(subdir, pref)
    
        self.logger.debug("Loading stats...")
        
        A = util.load("statsA", subdir, pref)
        self.stat_bestpath_times = A[0]
        self.stat_e_bp = A[1]
        self.stat_bestpath_R = A[2]
        
#         B = util.load("statsB", subdir, pref)
#         self.stat_m = B
#         
#         C = util.load("statsC", subdir, pref)
#         self.stat_e_1000 = C
    
    def report_stats(self, pref=""):
        self.test_agent.report_stats(pref="g_" + pref)
        
        util.plot([list(self.stat_bestpath_times), self.stat_bestpath_R],
                  self.stat_e_bp,
                  ["Finish time by best-action path",
                   "Reward for best-action path"],
                  title="Performance over time",
                  pref="bpt")
