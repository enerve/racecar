'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from environment import Environment
from car import Car
from track import CircleTrack
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
    log.configure_logger(logger, "Test RaceCar FS")
    logger.setLevel(logging.DEBUG)
    
    RADIUS = 98
    track = CircleTrack((0, 0), RADIUS, 20, Driver.NUM_JUNCTURES,
                        Environment.NUM_MILESTONES, Driver.NUM_LANES)
    NUM_SPEEDS = 3
    car = Car(Driver.NUM_DIRECTIONS, NUM_SPEEDS)

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

    environment = Environment(track, car, Driver.NUM_JUNCTURES,
                              should_record=True)
    trainer.drive_manual(environment, MY_IDEAL_A)
    
if __name__ == '__main__':
    test()
