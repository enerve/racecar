'''
Created on Sep 12, 2018

@author: enerve
'''

import logging
import math
import numpy as np

from track import Track

class CircleTrack(Track):
    
    
    def __init__(self, center, radius, width, num_milestones, num_lanes):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.center = center
        self.radius = radius
        self.width = width
        self.num_milestones = num_milestones
        self.num_lanes = num_lanes
        
    def get_starting_position(self):
        start_loc = (0, -self.radius)
        start_dir = 0  # in degrees, Right==0, Left==180
        return (start_loc, start_dir)
        
    def is_inside(self, location):
        (x, y) = location
        (a, b) = self.center
        outer_radius = self.radius + self.width / 2
        inner_radius = self.radius - self.width / 2
    
        d = math.sqrt((x - a) ** 2 + (y - b) ** 2)
        return d < outer_radius and d > inner_radius
    
    def progress_made(self, location):
        return self.theta(location) / (2 * math.pi)
    
    def theta(self, location):
        (x, y) = location
        (a, b) = self.center
        #d = math.sqrt((x - a) ** 2 + (y - b) ** 2)
        theta = math.atan2((y-b), (x-a)) + math.pi / 2 + 2 * math.pi
        if theta > 2 * math.pi:
            theta -= 2 * math.pi
        return theta

    def within_section(self, location, curr_milestone, next_milestone):
        assert next_milestone < self.num_milestones
        
        curr_milestone_theta = curr_milestone * 2 * math.pi / self.num_milestones
        next_milestone_theta = next_milestone * 2 * math.pi / self.num_milestones
        
        car_theta = self.theta(location)

        if curr_milestone_theta < next_milestone_theta:
            return car_theta < next_milestone_theta
        else:
            return car_theta < next_milestone_theta or \
                car_theta >= curr_milestone_theta
    
    def lane_encoding(self, location):
        (x, y) = location
        (a, b) = self.center
        inner_radius = self.radius - self.width / 2
    
        d = math.sqrt((x - a) ** 2 + (y - b) ** 2)
        return math.floor((d - inner_radius) * self.num_lanes / self.width)

    
    def draw(self):
        H = 220
        W = 240
        x = np.linspace(-W/2, W/2, W)
        y = np.linspace(-H/2, H/2, H).reshape(-1, 1)
        R_o = self.radius + self.width / 2
        R_i = self.radius - self.width / 2
        Circ = (x * x + y * y)
        color = 100
        A = color * np.logical_and(Circ < R_o**2, Circ > R_i**2)
        
        color = 250
        for m in range(self.num_milestones):
            rad = m * 2 * math.pi / self.num_milestones
            x = int(math.cos(rad) * R_i + H/2)
            y = int(math.sin(rad) * R_i + W/2)
            A[x, y] = color
            x = int(math.cos(rad) * R_o + H/2)
            y = int(math.sin(rad) * R_o + W/2)
            A[x, y] = color
        
        return A
