'''
Created on Sep 13, 2018

@author: enerve
'''

import logging
import math
import numpy as np

class Car():
    
    def __init__(self, num_directions, num_speeds):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # TODO :these should both be avoided (i.e. dir/speed may be real)
        self.num_directions = num_directions
        self.num_speeds = num_speeds
        
    def restart(self, track, should_record):
        self.location, self.direction = track.get_starting_position()
        self.speed = 1 # Default starting speed for this car type
        self.xy_step = None # cached x,y step tuple

        self.should_record = should_record
        self.action_history = []
        self.vector_history = []
        self.location_history = []

    def take_action(self, A):
        steer, accel = A
        # accel comes in as 0..2, which we convert to -1..1
        # steer comes in as 0..2 which we convert to -1..1
        self.accelerate(accel - 1)
        self.turn(steer - 1)
        if self.should_record:
            self.action_history.append(A)
            self.vector_history.append((self.direction, self.speed))

    def accelerate(self, accel):
        self.speed = self.speed + accel
        self.speed = max(self.speed, 1)
        self.speed = min(self.speed, self.num_speeds)
        self.xy_step = None

    def turn(self, steer):
        #   steer == -1 is left / anti-clockwise, +1 is right / clockwise
        dir_change = - steer * 360 // self.num_directions
        self.direction = (self.direction + dir_change + 360) % 360
        self.xy_step = None
        
    def move(self):
        ''' Move location for 1 time step
        '''
        if self.xy_step is None:
            dir_rad = math.radians(self.direction)
            x_step = self.speed * math.cos(dir_rad)
            y_step = self.speed * math.sin(dir_rad)
            self.xy_step = (x_step, y_step)

        (x, y) = self.location
        self.location = (x + self.xy_step[0], y + self.xy_step[1])
        
        if self.should_record:
            self.location_history.append(
                (self.location, self.speed, self.direction))
        
    def state_encoding(self):
        ''' Returns a simplified state tuple that agent can use.
        '''
        v = self.speed - 1 # v=0,1,2 for speed=1,2,3
        d = self.direction * self.num_directions // 360 
        #self.logger.debug("direction=%d .. d=%d", self.direction, d)
        return v, d

    def draw(self, coordinates, A):
        x, y = coordinates
        color = np.array([200, 0, 0])
        #self.logger.debug("%s %s", i, j)
        A[y, x] = color
