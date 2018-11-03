'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from car import Car
from track import CircleTrack
from trainer import Trainer
from driver import *
import cmd_line
import log
import numpy as np
import util


def main():
    args = cmd_line.parse_args()

    util.prefix_init(args)
    util.pre_problem = 'RC circle'

#     logger = logging.getLogger(__name__)
    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar FS")
    logger.setLevel(logging.DEBUG)
    
    # --------------
    
    NUM_JUNCTURES = 28
    NUM_MILESTONES = 27
    NUM_LANES = 5
    MAX_SPEED = NUM_SPEEDS = 3
    NUM_DIRECTIONS = 20
    
    NUM_STEER_POSITIONS = 3
    NUM_ACCEL_POSITIONS = 3

    RADIUS = 98
    WIDTH = 20
    track = CircleTrack((0, 0), RADIUS, WIDTH, NUM_JUNCTURES,
                        NUM_MILESTONES, NUM_LANES)
    car = Car(NUM_DIRECTIONS, NUM_SPEEDS)

    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   NUM_JUNCTURES:\t%s", NUM_JUNCTURES)
    logger.debug("   NUM_MILESTONES:\t%s", NUM_MILESTONES)
    logger.debug("   NUM_LANES:\t%s", NUM_LANES)
    logger.debug("   MAX_SPEED:\t%s", MAX_SPEED)
    logger.debug("   NUM_DIRECTIONS:\t%s", NUM_DIRECTIONS)
    logger.debug("   NUM_STEER_POSITIONS:\t%s", NUM_STEER_POSITIONS)
    logger.debug("   NUM_ACCEL_POSITIONS:\t%s", NUM_ACCEL_POSITIONS)
    
#     driver = QDriver(1, # alpha
#                     1, # gamma
#                     20, # explorate
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     driver = SarsaDriver(0.2, # alpha
#                     1, # gamma
#                     200, # explorate
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
    driver = SarsaLambdaDriver(0.9, # lambda
                    0.2, # alpha
                    1, # gamma
                    200, # explorate
                    NUM_JUNCTURES,
                    NUM_LANES,
                    NUM_SPEEDS,
                    NUM_DIRECTIONS,
                    NUM_STEER_POSITIONS,
                    NUM_ACCEL_POSITIONS)
    trainer = Trainer(driver, track, car)
    #subdir = "RC rect_qlearn_100_1.0_1.0_363312_"
    subdir = None
    if subdir:
        driver.load_model(subdir)
        trainer.load_stats(subdir)
    trainer.train(15*1000)
    trainer.report_stats()
    
    trainer.play_best()
    best_env = trainer.best_environment()
    debug_driver = ManualDriver(NUM_JUNCTURES,
                                NUM_LANES,
                                NUM_SPEEDS,
                                NUM_DIRECTIONS,
                                NUM_STEER_POSITIONS,
                                NUM_ACCEL_POSITIONS,
                                best_env.get_action_history())
    debug_driver.run_episode(track, car)

    # --------- CV ---------
#     import random
#     lambdas = []
#     alphas = []
#     scores = []
#     expls = []
# 
# 
#     for rep in range(20):
#         lam = random.random() 
#         alp = random.uniform(0.3, 1.0)
#         expl = 10 ** random.uniform(1.0, 3)
#         logger.debug("--- rep %d --- lam: %0.2f, alp: %0.2f, expl: %0.2f", 
#                      rep, lam, alp, expl)
#         
#         driver = QLambdaDriver(lam, # lambda
#                         alp, # alpha
#                         1, # gamma
#                         expl, # explorate
#                         NUM_JUNCTURES,
#                         NUM_LANES,
#                         NUM_SPEEDS,
#                         NUM_DIRECTIONS,
#                         NUM_STEER_POSITIONS,
#                         NUM_ACCEL_POSITIONS)
#         trainer = Trainer(driver, track, car)
#         bp_times, e_bp, bp_R, bp_j = trainer.train(20*1000, save_to_file=False,
#                                              seed = 513 + rep)
#         #bp_R=[random.randrange(20, 1000)]
#                 
#         lambdas.append(lam)
#         alphas.append(alp)
#         expls.append(expl)
#                
#         score = 0
#         for i in range(5):
#             score += bp_R[-1-i] / bp_j[-1-i]
#         score /= 5
#         scores.append(score)
#         logger.debug("  Score: %s", score)
#             
#  
#  
#         logger.debug("lambdas.extend( %s)", lambdas)
#         logger.debug("alphas.extend(  %s)", alphas)
#         logger.debug("expls.extend(   %s)", expls)
#         logger.debug("scores.extend(  %s)", scores)
#        
#     util.scatter(np.array(lambdas), np.array(alphas), np.array(scores),
#                  "lambda", "alpha", pref="cv_la")
#     util.scatter(np.array(lambdas), np.array(expls), np.array(scores),
#                  "lambda", "expl", pref="cv_le")
#     util.scatter(np.array(expls), np.array(alphas), np.array(scores),
#                  "expl", "alpha", pref="cv_ea")


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
