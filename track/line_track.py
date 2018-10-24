'''
Created on Sep 14, 2018

@author: enerve
'''

from functools import lru_cache
import logging
import math
import numpy as np

from track import Track

class LineTrack(Track):
    '''
    A track defined by a loop of connected lines.
    '''

    def __init__(self, points, track_width, num_junctures, num_milestones,
                 num_lanes):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # The sequence of points that when connected with lines defines the
        #     loop representing the center of the track.
        self.points = points
        
        self.W, self.H = 240, 220
        self.track_width = track_width
        self.num_junctures = num_junctures
        self.num_milestones = num_milestones
        self.num_lanes = num_lanes

        self.lines = []
        self.lengths = []
        self.cummu_lengths = []
        self.total_length = 0
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i+1) % len(self.points)]
            np1, np2 = np.array(p1), np.array(p2)
            self.lines.append((np1, np2))
            
            length = np.sum((np1 - np2) ** 2)
            self.lengths.append(length)
            self.cummu_lengths.append(self.total_length)
            self.total_length += length

    def get_starting_position(self):
        return (self.points[0], 0) # going right
        
    def leftness_ratio(self, c, line, dist):
        ''' Returns how near the given location is to the left edge of the
            track. Returns (0..1)
        '''
        radius = self.track_width / 2
        rotline = (-line[1], line[0]) # line rotated clockwise
        leftness = -dist if np.inner(c, rotline) < 0 else dist
        return (leftness + radius) / self.track_width

    def anchor(self, loc, prev_anchor=None):
        ''' Extracts information about the given location and returns a tuple:
                fraction of track covered by this location,
                fraction of distance from "inner edge" of road,
                the id of the line nearest to this location (if it's nearest),
                else, the id of the point/corner nearest to this location
        '''
        c_ = np.array(loc)
        radius = self.track_width / 2
        anchor = None
        
        # Loop over each line and find the best anchor point for loc
        if prev_anchor:
            min_i = prev_anchor['line_id']
            min_i = prev_anchor['point_id'] if min_i is None else 0
        else:
            min_i = 0
        nearest_line = None
        nearest_sqdist = 0
        for i, (p1_, p2_) in enumerate(self.lines[min_i:], start=min_i):
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
                            (self.cummu_lengths[i] + \
                             fraction_of_line * self.lengths[i]) / \
                             self.total_length
                        dist = math.sqrt(sqdist)
                        anchor = {'progress': nearest_progress,
                                  'leftness': self.leftness_ratio(c, line, dist),
                                  'line_id': i,
                                  'point_id': None}
            if nearest_line is not None and nearest_line < i:
                # Nearest line is behind us already, so
                # we're not going to find anything better
                break;

            if nearest_line is None:
                # check if location is near corner p2_
                c2 = c_ - p2_
                dist = np.inner(c2, c2)
                if dist <= radius:
                    # TODO: progres within curve?
                    anchor = {'progress': self.cummu_lengths[i] / self.total_length,
                              'leftness': self.leftness_ratio(c2, line, dist),
                              'line_id': None,
                              'point_id': i}
                    # TODO: Note that we're assuming Finish line is along a
                    #        STRAIGHT road, not in a curve
                    break


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
        W, H = self.W, self.H
        x = np.linspace(-x, W-x, W)
        y = np.linspace(-y, H-y, H).reshape(-1, 1)
        return x**2 + y**2 <= radius**2

    @lru_cache(maxsize=None)
    def draw_matrix(self, draw_lanes=False):
        W, H = self.W, self.H
        A = np.zeros((H, W))
        
        multipoints = []
        for (p1_, p2_) in self.lines:
            x1, x2 = p1_[0], p2_[0]
            y1, y2 = self.H - p1_[1], self.H - p2_[1]
            d = max(abs(x2 - x1), abs(y2 - y1))
            for j in range(d):
                x = int(x1 + j * (x2 - x1) / d)
                y = int(y1 + j * (y2 - y1) / d)
                multipoints.append((x, y))
        
        for x, y in multipoints:
            C = self.circle_splotch(x, y, radius = self.track_width / 2)
            A = np.logical_or(A, C)

        A = A * 50
        A2 = np.zeros((H, W, 3), dtype=np.uint8)
        A2 = (A.T + A2.T).T
        A2 = A2 + (255 - 50)

        if draw_lanes:
            color = np.array([0, 150, 250])
            j_length = self.total_length / self.num_junctures
            progress = 0
            l = 0
            for j in range(self.num_junctures):
                progress += j_length
                
                while l < len(self.lengths):
                    if self.lengths[l] > progress:
                        break
                    progress -= self.lengths[l]
                    l += 1
                else:
                    # use the end of the last line
                    l -= 1
                    progress = self.lengths[l]
                    
                x1, y1, x2, y2 = self.lines[l]
                p1_ = np.array((x1, y1))
                p2_ = np.array((x2, y2))
                line = p2_ - p1_
                proj_ = p1_ + line * progress / self.lengths[l]
                
                rotline = np.array((-line[1], line[0])) # line rotated clockwise
                rotline = rotline / self.lengths[l]
                rotline = rotline * self.track_width
                for n in range(self.num_lanes):
                    x_ = np.round(proj_ + rotline * ((n+0.5) / self.num_lanes - 0.5))
                    #self.logger.debug(
                    #    "j%d: x_%s   proj %s plus rotline %s... from p1 %s to p2 %s... progress %0.2f / length %0.2f",
                    #    j, x_, proj_, rotline, p1_, p2_, progress, self.lengths[l])
                    c_ = self.location_to_coordinates(x_)
                    A2[c_[1], c_[0]] = color

        return A2
        
    def draw(self):
        return self.draw_matrix().copy()
