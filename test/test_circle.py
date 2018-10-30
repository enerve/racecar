'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from environment import Environment
from car import Car
from track import CircleTrack
from manual_driver import ManualDriver

import cmd_line
import log
import util


def test():

    args = cmd_line.parse_args()

    util.prefix_init(args)
    util.pre_problem = 'RC'

    logger = logging.getLogger()
    log.configure_logger(logger, "Test RaceCar FS")
    logger.setLevel(logging.DEBUG)
    
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
    NUM_SPEEDS = 3
    car = Car(NUM_DIRECTIONS, NUM_SPEEDS)

    MY_IDEAL_A = [
            (0, 2),
            (1, 2),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ]

    environment = Environment(track, car, NUM_JUNCTURES,
                              should_record=True)
    driver = ManualDriver(
        NUM_JUNCTURES,
        NUM_LANES,
        NUM_SPEEDS,
        NUM_DIRECTIONS,
        NUM_STEER_POSITIONS,
        NUM_ACCEL_POSITIONS,
        MY_IDEAL_A)
    total_R, environment = driver.run_episode(track, car)
    logger.debug("Total_R: %s", total_R)
    environment.report_history()
    environment.play_movie(save=False, pref="bestmovie")
    
if __name__ == '__main__':
    test()
