'''
Created on Sep 14, 2018

@author: enerve
'''

import logging
import matplotlib.pyplot as plt
import math
import numpy as np

from track import Track

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

class LineTrack(Track):
    '''
    A track defined by a loop of connected lines.
    '''

    def __init__(self, points, width, num_junctures, num_milestones,
                 num_lanes):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # The sequence of points that when connected with lines defines the
        #     loop representing the center of the track.
        self.points = points
        
        self.W = 240
        self.H = 220
        self.width = width
        self.num_junctures = num_junctures
        self.num_milestones = num_milestones
        self.num_lanes = num_lanes

        self.track_matrix = None

        self.lines = []
        self.lengths = []
        self.total_length = 0
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i+1) % len(self.points)]
            x1, y1 = p1
            x2, y2 = p2
            self.lines.append((x1, y1, x2, y2))
            
            length = math.sqrt((x1-x2)**2 + (y1-y2)**2)
            self.lengths.append(length)
            self.total_length += length

    def get_starting_position(self):
        return (self.points[0], 0) # going right
        
    def leftness_ratio(self, c, line, dist):
        radius = self.width / 2
        rotline = (-line[1], line[0]) # line rotated clockwise
        leftness = -dist if np.inner(c, rotline) < 0 else dist
        return (leftness + radius) / self.width

    def anchor(self, loc):
        ''' Extracts information about the given location and returns a tuple:
                fraction of track covered by this location,
                fraction of distance from "inner edge" of road,
                the id of the line nearest to this location (if it's nearest),
                else, the id of the point/corner nearest to this location
        '''
        c_ = np.array(loc)
        radius = self.width / 2
        anchor = None
        
        # Check if distance from one of the lines is within radius
        cummu_len = 0
        nearest_line = None
        nearest_sqdist = 0
        for i, (x1, y1, x2, y2) in enumerate(self.lines):
            p1_ = np.array((x1, y1))
            p2_ = np.array((x2, y2))
            line = p2_ - p1_
            c = c_ - p1_
            
            fraction_of_line = np.inner(c, line) / np.inner(line, line)
            if fraction_of_line >= 0 and fraction_of_line <= 1:
                proj_ = p1_ + line * fraction_of_line
                sqdist = np.inner(c_ - proj_, c_ - proj_)
                if sqdist <= radius ** 2:
                    # We're close enough to this line that it counts
                    if nearest_line is None or nearest_sqdist > sqdist:
                        nearest_sqdist = sqdist 
                        nearest_line = i
                        nearest_progress = \
                            (cummu_len + fraction_of_line * self.lengths[i]) / \
                            self.total_length
                        dist = math.sqrt(sqdist)
                        anchor = {'progress': nearest_progress,
                                  'leftness': self.leftness_ratio(c, line, dist),
                                  'line_id': i,
                                  'point_id': None}
            if nearest_line is not None and nearest_line < i:
                # we're not gonna find anything better
                break;

            if nearest_line is None:
                # check if location is near corner p1_
                dist = np.inner(c, c)
                if dist <= radius:
                    # TODO: progres within curve?
                    anchor = {'progress': cummu_len / self.total_length,
                              'leftness': self.leftness_ratio(c, line, dist),
                              'line_id': None,
                              'point_id': i}
                    # TODO: Note that we're assuming finish line is along a
                    #        STRAIGHT road, not in a curve
                    break

            cummu_len += self.lengths[i]

        # If anchor is still None, the car is out of bounds / has crashed

        return anchor

    def is_inside(self, anchor):
        return anchor is not None

    def progress_made(self, anchor):
        return anchor['progress']

    def within_juncture(self, anchor, juncture_a, juncture_b):
        assert juncture_b < self.num_junctures

        j = math.floor(self.progress_made(anchor) * self.num_junctures)
    
        if juncture_a < juncture_b:
            return j < juncture_b
        else:
            return j < juncture_b or \
                j >= juncture_a

    def within_milestone(self, anchor, milestone_a, milestone_b):
        assert milestone_b < self.num_milestones

        m = math.floor(self.progress_made(anchor) * self.num_milestones)

        if milestone_a < milestone_b:
            return m < milestone_b
        else:
            return m < milestone_b or \
                m >= milestone_a
    
    def lane_encoding(self, anchor):
        return math.floor(anchor['leftness'] * self.num_lanes)

    def location_to_coordinates(self, location):
        ''' Return image coordinates for real-world geometric location
        '''
        x, y = location
        return (int(x), int(-y))

    def coordinates_to_location(self, coordinates):
        ''' Return real-world geometric location for given image coordinates
        '''
        x_, y_ = coordinates
        return (x_, -y_)
    
    def circle_splotch(self, x, y, radius):
        W = self.W
        H = self.H
        x = np.linspace(-x, W-x, W)
        y = np.linspace(-y, H-y, H).reshape(-1, 1)
        Circ = (x * x + y * y)
        return Circ <= radius**2

    def draw_matrix(self):
        W = self.W
        H = self.H
        A = np.zeros((H, W))
        
        multipoints = []
        for (x1, y1, x2, y2) in self.lines:
            y1 = self.H - y1
            y2 = self.H - y2
            d = max(abs(x2 - x1), abs(y2 - y1))
            for j in range(d):
                x = int(x1 + j * (x2 - x1) / d)
                y = int(y1 + j * (y2 - y1) / d)
                multipoints.append((x, y))
        
        radius = self.width / 2
        for p in multipoints:
            x, y = p
            C = self.circle_splotch(x, y, radius)
            A = np.logical_or(A, C)

        A = A * 50
        A2 = np.zeros((H, W, 3), dtype=np.uint8)
        A2 = (A.T + A2.T).T
        A2 = A2 + (255 - 50)

#         # Draw junctures/milestones
#         for (x1, y1, x2, y2) in self.lines:
#             
#             lengths
#             d = max(abs(x2 - x1), abs(y2 - y1))
#             for j in range(d):
#                 x = int(x1 + j * (x2 - x1) / d)
#                 y = int(y1 + j * (y2 - y1) / d)
#                 multipoints.append((x, y))

        color = np.array([10, 10, 10])
        for (x, y) in self.points:
            A2[self.H - y, x] = color

        return A2
        
    def draw(self):
        if self.track_matrix is None:
            self.track_matrix = self.draw_matrix()
            
        A = self.track_matrix.copy()
            
        return A
