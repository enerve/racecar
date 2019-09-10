'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from car import Car
from driver import *
from function import *
from track import LineTrack
from trainer import Trainer
import cmd_line
import log
import trainer_helper as th
import util

import numpy as np
import random
import matplotlib.pyplot as plt

def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'RC X'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar X")
    logger.setLevel(logging.DEBUG)

    # --------------
    
    points = [
            (45, 10),
            (80, 10),
            (120, 90),
            (160, 10),
            (230, 10),
            (195, 110),
            (230, 210),
            (160, 210),
            (120, 130),
            (80, 210),
            (10, 210),
            (45, 110),
            (10, 10)
        ]

    config = th.CONFIG(
        NUM_JUNCTURES = 200,
        NUM_MILESTONES = 200,
        NUM_LANES = 5,
        NUM_SPEEDS = 3,
        NUM_DIRECTIONS = 20,
        NUM_STEER_POSITIONS = 3,
        NUM_ACCEL_POSITIONS = 3
    )

    WIDTH = 20
    track = LineTrack(points, WIDTH, config)
    car = Car(config)
    
    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   %s", config)
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    # ------------------ Guide driver FA -----------------

    driver_fa = QLookup(config,
                        alpha=0.4)

    # ------------------ Mimic FA -----------------
    mimic_fa = None

    # ------------------ Guide driver RL algorithm ---------------

    driver = th.create_driver(config, 
                    alg = 'sarsalambda', 
                    expl = 19,
                    lam = 0.9,
                    fa=driver_fa,
                    mimic_fa=mimic_fa)

    # ------------------ Student driver FA -----------------

    # ------------------ Student driver RL algorithm -------------
    student = None

    # ------------------ Training -------------------

    trainer = Trainer(driver, track, car)
    
#     subdir = "212870_RC X_sarsa_lambda_400_0.2_1.0_0.9_"
#     driver.load_model(subdir)
#     trainer.load_stats(subdir)
    trainer.train(100*1000)

    trainer.report_stats()
             

    t_R, b_E, _ = driver.run_best_episode(track, car, False)
    logger.debug("Driver best episode total R = %0.2f time=%d", t_R,
                 b_E.total_time_taken())
    b_E.play_movie(pref="bestmovie")

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
    
if __name__ == '__main__':
    main()
