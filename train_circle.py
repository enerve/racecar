'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from car import Car
from track import CircleTrack
from q_driver import QDriver
import cmd_line
import log
import util
import trainer


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
    track = CircleTrack((0, 0), RADIUS, 20, NUM_JUNCTURES,
                        NUM_MILESTONES, NUM_LANES)
    car = Car(NUM_DIRECTIONS, NUM_SPEEDS)
     
     
    #original_driver = QDriver(alpha=1, gamma=1, explorate=2500)
    Q_filename = "RC_qlearn_341830_Q_50__.csv"
    driver = QDriver(1, # alpha
                    1, # gamma
                    20, # explorate
                    NUM_JUNCTURES,
                    NUM_LANES,
                    NUM_SPEEDS,
                    NUM_DIRECTIONS,
                    NUM_STEER_POSITIONS,
                    NUM_ACCEL_POSITIONS,
                    load_Q_filename=Q_filename)
    #driver = Driver(alpha=0.2, gamma=1, load_filename="RC_qlearn_652042_Q_28_.csv")
    #trainer.train(driver, track, car, 400*1000)
    #trainer.play_best(driver, track, car)
    driver.report_stats("static")
             

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
