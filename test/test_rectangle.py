'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from environment import Environment
from car import Car
from track import LineTrack
from driver import Driver
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
    WIDTH = 20
    track = LineTrack(points, WIDTH, NUM_JUNCTURES, Environment.NUM_MILESTONES,
                      Driver.NUM_LANES)
    #track.draw()
    
    NUM_SPEEDS = 3
    car = Car(Driver.NUM_DIRECTIONS, NUM_SPEEDS)

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
