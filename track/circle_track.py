'''
Created on Sep 12, 2018

@author: enerve
'''

import logging
import math
import numpy as np

from track import Track

class CircleTrack(Track):
    
    
    def __init__(self, center, radius, width, num_junctures, num_milestones,
                 num_lanes):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug("Init CircleTrack")

        self.H = 220
        self.W = 240
        self.center = center
        self.radius = radius
        self.width = width
        self.num_junctures = num_junctures
        self.num_milestones = num_milestones
        self.num_lanes = num_lanes
        
    def get_starting_position(self):
        start_loc = (0, -self.radius)
        start_dir = 0  # in degrees, Right==0, Left==180
        return (start_loc, start_dir)

    def anchor(self, location):
        (x, y) = location
        (a, b) = self.center
        d = math.sqrt((x - a) ** 2 + (y - b) ** 2)

        theta = self.theta(location)
        progress = theta / (2 * math.pi)
        return {
            'center_dist': d,
            'theta': theta,
            'progress': progress
            }
        
    def is_inside(self, anchor):
        outer_radius = self.radius + self.width / 2
        inner_radius = self.radius - self.width / 2
        d = anchor['center_dist']
        return d < outer_radius and d > inner_radius
    
    def progress_made(self, anchor):
        return anchor['progress']
    
    def theta(self, location):
        (x, y) = location
        (a, b) = self.center
        theta = math.atan2((y-b), (x-a)) + math.pi / 2 + 2 * math.pi
        if theta > 2 * math.pi:
            theta -= 2 * math.pi
        return theta

    def within_juncture(self, anchor, curr_juncture, next_juncture):
        assert next_juncture < self.num_junctures
        
        curr_juncture_theta = curr_juncture * 2 * math.pi / self.num_junctures
        next_juncture_theta = next_juncture * 2 * math.pi / self.num_junctures
        
        car_theta = anchor['theta']

        if curr_juncture_theta < next_juncture_theta:
            return car_theta < next_juncture_theta
        else:
            return car_theta < next_juncture_theta or \
                car_theta >= curr_juncture_theta

    def within_milestone(self, anchor, curr_milestone, next_milestone):
        assert next_milestone < self.num_milestones
        
        curr_milestone_theta = curr_milestone * 2 * math.pi / self.num_milestones
        next_milestone_theta = next_milestone * 2 * math.pi / self.num_milestones
        
        car_theta = anchor['theta']

        if curr_milestone_theta < next_milestone_theta:
            return car_theta < next_milestone_theta
        else:
            return car_theta < next_milestone_theta or \
                car_theta >= curr_milestone_theta
    
    def lane_encoding(self, anchor):
        inner_radius = self.radius - self.width / 2
        d = anchor['center_dist']
        return math.floor((d - inner_radius) * self.num_lanes / self.width)

    def location_to_coordinates(self, location):
        ''' Return image coordinates for real-world geometric location
        '''
        H = self.H
        W = self.W
        x, y = location
        x_ = int(x + W / 2)
        y_ = int(-y + H / 2)
        return x_, y_

    def coordinates_to_location(self, coordinates):
        ''' Return real-world geometric location for given image coordinates
        '''
        H = self.H
        W = self.W
        x_, y_ = coordinates
        x = x_ - W / 2
        y = H/2 - y_
        return x, y
        
    def draw(self):
        H = self.H
        W = self.W
        x = np.linspace(-W/2, W/2, W)
        y = np.linspace(-H/2, H/2, H).reshape(-1, 1)
        R_o = self.radius + self.width / 2
        R_i = self.radius - self.width / 2
        Circ = (x * x + y * y)
        A = 50 * np.logical_and(Circ < R_o**2, Circ > R_i**2)
        A2 = np.zeros((H, W, 3), dtype=np.uint8)
        A2 = (A.T + A2.T).T
        A2 = A2 + 255 - 50
        
        color = [10, 10, 250]
        for j in range(self.num_junctures):
            rad = j * 2 * math.pi / self.num_junctures
            x = int(math.cos(rad) * R_i + H/2)
            y = int(math.sin(rad) * R_i + W/2)
            A2[x, y] = color
            x = int(math.cos(rad) * R_o + H/2)
            y = int(math.sin(rad) * R_o + W/2)
            A2[x, y] = color
        
        return A2
