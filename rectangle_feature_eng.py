'''
Created on 14 Feb 2019

@author: enerve
'''

import logging
import math
import torch
import util

from function import FeatureEng

class RectangleFeatureEng(FeatureEng):
    '''
    Feature engineering for polynomial regression of rectangular track
    '''

    def __init__(self, num_junctures,
                 num_lanes,
                 num_speeds,
                 num_directions,
                 num_steer_positions,
                 num_accel_positions,
                 include_basis = False,
                 include_sin_cosine = False,
                 include_splines = False,
                 spline_length = 0,
                 include_corner_splines = False,
                 corners = [],
                 include_bounded_features = False,
                 poly_degree = 1):
        '''
        Constructor
        '''
        
        # states
        self.num_junctures = num_junctures
        self.num_lanes = num_lanes
        self.num_speeds = num_speeds
        self.num_directions = num_directions
        # actions
        self.num_steer_positions = num_steer_positions
        self.num_accel_positions = num_accel_positions
        # feature engineering options
        self.include_basis = include_basis
        self.include_sin_cosine = include_sin_cosine
        self.include_splines = include_splines
        self.spline_length = spline_length
        self.include_corner_splines = include_corner_splines
        self.corners = corners
        self.include_bounded_features = include_bounded_features
        self.poly_degree = poly_degree

        self.device = torch.device('cuda' if util.use_gpu else 'cpu')
        
        # Hard-coded normalization params for the RaceCar state parameters        
        shift_list = []
        scale_list = []

        if self.include_basis:
            shift_list.append(0)
            scale_list.append(1)
        
        shift_list.extend([
                     num_lanes / 2.,
                     num_speeds / 2.,
                     num_directions / 2.])
        scale_list.extend([
                     num_lanes,
                     num_speeds,
                     num_directions])

        if include_splines:
            for knot in range(spline_length, num_junctures,
                              spline_length):
                shift_list.append(knot / 2)
                scale_list.append(knot)
        else:
            shift_list.append(num_junctures / 2.)
            scale_list.append(num_junctures)           

        if include_corner_splines:
            for ka, kb in zip(corners, corners[1:]):
                shift_list.append((ka + kb) / 2)
                scale_list.append(kb - ka)

        if include_sin_cosine:
            shift_list.extend([0,
                         0,
                         0,
                         0])
            scale_list.extend([1,
                         1,
                         1,
                         1])
                
        if include_bounded_features:
            shift_list.extend([1,
                         1])
            scale_list.extend([2,
                         2])

        self.shift = torch.Tensor(shift_list).to(self.device)
        self.scale = torch.Tensor(scale_list).to(self.device)
        self.num_inputs = len(shift_list)
        
    def num_actions(self):
        return self.num_steer_positions * self.num_accel_positions
        
    def prefix(self):
        return '%d%s%s%s%s' % (self.poly_degree,
                             't' if self.include_sin_cosine else 'f',
                             't' if self.include_splines else 'f',
                             't' if self.include_bounded_features else 'f',
                             't' if self.include_corner_splines else 'f')

    def initial_W(self):
        return torch.randn(self.num_inputs ** self.poly_degree,
                           self.num_actions()).to(self.device)

    def x_adjust(self, juncture, lane, speed, direction):
        ''' Takes the input params and converts it to an input feature array
        '''

        x_list = []
        
        if self.include_basis:
            x_list.append(1)
        
        x_list.extend([lane, speed, direction])

        if self.include_splines:
            for knot in range(self.spline_length, self.num_junctures,
                              self.spline_length):
                #x_list.append(max(0, -(juncture - knot)))
                
                ka = knot - self.spline_length
                kb = knot
                x_list.append(ka if juncture < ka else
                              kb if juncture > kb else
                              juncture)
        else:
            x_list.append(juncture)
            
        # corners at(after) juncture 4
        if self.include_corner_splines:
            for ka, kb in zip(self.corners, self.corners[1:]):
                x_list.append(ka if juncture < ka else
                              kb if juncture > kb else
                              juncture)
        
        if self.include_sin_cosine:
            # feature engineering
            sj = math.sin(juncture * 2 * math.pi / self.num_junctures)
            cj = math.cos(juncture * 2 * math.pi / self.num_junctures)
            sd = math.sin(direction * 2 * math.pi / self.num_directions)
            cd = math.cos(direction * 2 * math.pi / self.num_directions)
            x_list.extend([sj, cj, sd, cd])
            
        if self.include_bounded_features:
            x_list.append(max(0, lane - 1))
            x_list.append(max(0, 3 - lane))

        x1 = torch.Tensor(x_list).to(self.device)
        x1 -= self.shift
        x1 /= self.scale

        x = x1
        if self.poly_degree >= 2:
            x = torch.ger(x, x1).flatten()
            if self.poly_degree == 3:
                x = torch.ger(x, x1).flatten()
            elif self.poly_degree == 4:
                x = torch.ger(x, x).flatten()

        return x
    
    def a_index(self, a_tuple):
        steer, accel = a_tuple
        return self.num_accel_positions * steer + accel

    def a_tuple(self, a_index):
        return (a_index // self.num_accel_positions,
                a_index % self.num_accel_positions)


