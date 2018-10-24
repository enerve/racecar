'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from car import Car
from q_driver import QDriver
from track import LineTrack
from trainer import Trainer
import cmd_line
import log
import util

def main():
    args = cmd_line.parse_args()

    util.prefix_init(args)
    util.pre_problem = 'RC rect'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar FS rectangle")
    logger.setLevel(logging.DEBUG)

    # --------------
    
    points = [
            (120, 40),
            (210, 40),
            (210, 180),
            (30, 180),
            (30, 40)
        ]
    NUM_JUNCTURES = 50
    NUM_MILESTONES = 50
    NUM_LANES = 5
    MAX_SPEED = NUM_SPEEDS = 3
    NUM_DIRECTIONS = 20
    
    NUM_STEER_POSITIONS = 3
    NUM_ACCEL_POSITIONS = 3

    WIDTH = 20
    track = LineTrack(points, WIDTH, NUM_JUNCTURES, NUM_MILESTONES,
                      NUM_LANES)
    
    car = Car(NUM_DIRECTIONS, NUM_SPEEDS)
    
    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   NUM_JUNCTURES:\t%s", NUM_JUNCTURES)
    logger.debug("   NUM_MILESTONES:\t%s", NUM_MILESTONES)
    logger.debug("   NUM_LANES:\t%s", NUM_LANES)
    logger.debug("   MAX_SPEED:\t%s", MAX_SPEED)
    logger.debug("   NUM_DIRECTIONS:\t%s", NUM_DIRECTIONS)
    logger.debug("   NUM_STEER_POSITIONS:\t%s", NUM_STEER_POSITIONS)
    logger.debug("   NUM_ACCEL_POSITIONS:\t%s", NUM_ACCEL_POSITIONS)
    
    driver = QDriver(1, # alpha
                    1, # gamma
                    5000, # explorate
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
    trainer.train(2*1000)
    trainer.report_stats()
    trainer.play_best()
        

    # --------- CV ---------
#     explorates = [50, 100, 200, 400, 800]
#     stats_bp_times = []
#     stats_e_bp = []
#     stats_labels = []
#     num_episodes = 60 * 1000
#     for explorate in explorates:
#         logger.debug("--- Explorate=%d ---" % explorate)
#         #for i in range(3):
#         i=0
#         seed = (100 + 53*i)
#         pref = "%d_%d" % (explorate, seed)
#         driver = QDriver(1, # alpha
#                         1, # gamma
#                         explorate, # explorate
#                         NUM_JUNCTURES,
#                         NUM_LANES,
#                         NUM_SPEEDS,
#                         NUM_DIRECTIONS,
#                         NUM_STEER_POSITIONS,
#                         NUM_ACCEL_POSITIONS)
#         stat_bestpath_times, stat_e_bp = \
#             trainer.train(driver, track, car, num_episodes, seed=seed, pref=pref)
#         stats_bp_times.append(stat_bestpath_times)
#         stats_e_bp.append(stat_e_bp)
#         stats_labels.append("N0=%d seed=%d" % (explorate, seed))
#         logger.debug("bestpath: %s", stat_bestpath_times)
#         logger.debug("stat_e: %s", stat_e_bp)
#         trainer.play_best(driver, track, car, should_play_movie=False,
#                           pref=pref)
#     util.plot_all(stats_bp_times, stats_e_bp, stats_labels,
#                   title="Time taken by best path as of epoch", pref="BestTimeTaken")
    
if __name__ == '__main__':
    main()
