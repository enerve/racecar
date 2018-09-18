'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from environment import Environment
from car import Car
from track import LineTrack
import trainer

import cmd_line
import log
import util


def test():

    args = cmd_line.parse_args()

    util.prefix_init(args)
    util.pre_problem = 'RC'

    logger = logging.getLogger()
    log.configure_logger(logger, "Test RaceCar FS rectangle")
    logger.setLevel(logging.DEBUG)
    
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
    #track.draw()
    
    NUM_SPEEDS = 3
    car = Car(NUM_DIRECTIONS, NUM_SPEEDS)

    MY_IDEAL_A = [
            (1, 2),
            (1, 2),
            (1, 1),
            (1, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
            (1, 1),
        ]
#     logger.debug("About to drive manual")

    environment = Environment(track, car, NUM_JUNCTURES,
                              should_record=True)
    trainer.drive_manual(environment, MY_IDEAL_A)
    
if __name__ == '__main__':
    test()
