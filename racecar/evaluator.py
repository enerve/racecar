'''
Created on 23 May 2019

@author: enerve
'''

import numpy as np

import logging
from really import util
from really.evaluator import Evaluator as Eval

class Evaluator(Eval):
    
    def __init__(self, config, episode_factory, test_agent, fa):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.config = config
        self.episode_factory = episode_factory
        self.test_agent = test_agent
        self.fa = fa

        # track #steps/time-taken by bestpath, as iterations progress
        self.stat_bestpath_times = []
        # track recent average of rewards collected, as iterations progress
        self.stat_bestpath_R = []
        self.stat_bestpath_juncture = []
        self.stat_e_bp = []

        self.q_plotter = util.Plotter("Q value at junctures along lanes")
#         self.c_plotter = util.Plotter("C value at junctures along lanes")

        self.bestpath_plotter = util.Plotter("Best path")

    def evaluate(self, ep, epoch): #TODO: rename to collect_stats??
        episode = self.episode_factory.new_episode([self.test_agent])
        episode.start_recording()
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
        
        Q = np.zeros((self.config.NUM_JUNCTURES,
                      self.config.NUM_LANES,
                      self.config.NUM_SPEEDS,
                      self.config.NUM_DIRECTIONS))
        for j in range(self.config.NUM_JUNCTURES):
            for l in range(self.config.NUM_LANES):
                for s in range(self.config.NUM_SPEEDS):
                    for d in range(self.config.NUM_DIRECTIONS):
                        S = (j, l, s, d)
                        ai, v, V = self.fa.best_action(S)
                        #self.fa.feature_eng.
                        Q[j, l, s, d] = v
        self.q_plotter.add_image(self._plottable(Q, (2, 3), True),
                                        "Epoch %d" % epoch)

#         self.c_plotter.add_image(self._plottable(self.C, (2, 3, 4, 5)))

        self.bestpath_plotter.add_image(episode.path_image(),
                                        "Epoch %d" % epoch)

    def _plottable(self, X, axes, pick_max=False):
        X = np.max(X, axis=axes) if pick_max else np.sum(X, axis=axes)
        return X.reshape(X.shape[0], -1).T

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
        self.stat_bestpath_times = list(A[0])
        self.stat_e_bp = list(A[1])
        self.stat_bestpath_R = list(A[2])
        
#         B = util.load("statsB", subdir, pref)
#         self.stat_m = B
#         
#         C = util.load("statsC", subdir, pref)
#         self.stat_e_1000 = C
    
    def report_stats(self, pref=""):
        
        self.q_plotter.play_animation(cmap='hot', interpolation='nearest',
                                      interval=750,
                                      show_axis=True, pref="QLanes")
        
        self.bestpath_plotter.play_animation(aspect='equal', interval=500,
                                             pref="bestpath")
        
        self.test_agent.report_stats(pref="g_" + pref)
        
        util.plot([list(self.stat_bestpath_times), self.stat_bestpath_R],
                  self.stat_e_bp,
                  ["Finish time by best-action path",
                   "Reward for best-action path"],
                  title="Performance over time",
                  pref="bpt")
