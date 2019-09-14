'''
Created on 12 Mar 2019

@author: enerve
'''

import math
import torch

from .racecar_feature_eng import RacecarFeatureEng

class CircleSAFeatureEng(RacecarFeatureEng):
    '''
    Feature engineering for States and Actions of circular track
    '''

    INCLUDE_SIN_COSINE = True
    SPLINE = True
    SPLINE_LENGTH = 4
    BOUNDED_FEATURES = True
    POLY_DEGREE = 3

    def __init__(self,
                 config):

        super().__init__(config)

        # Hard-coded normalization params for the RaceCar state parameters
        
        shift_list = [0,
                    self.num_lanes / 2.,
                    self.num_speeds / 2.,
                    self.num_directions / 2.,
                    self.num_steer_positions / 2.,
                    self.num_accel_positions / 2.]
        scale_list = [1,
                    self.num_lanes,
                    self.num_speeds,
                    self.num_directions,
                    self.num_steer_positions,
                    self.num_accel_positions]

        if self.SPLINE:
            for knot in range(self.SPLINE_LENGTH, self.num_junctures,
                              self.SPLINE_LENGTH):
                shift_list.append(knot / 2)
                scale_list.append(knot)
        else:
            shift_list.append(self.num_junctures / 2.)
            scale_list.append(self.num_junctures)           

        if self.INCLUDE_SIN_COSINE:
            shift_list.extend([0,
                         0,
                         0,
                         0])
            scale_list.extend([1,
                         1,
                         1,
                         1])
                
        if self.BOUNDED_FEATURES:
            shift_list.extend([1,
                         1,
                         0.5,
                         0.5,
                         0.5])
            scale_list.extend([2,
                         2,
                         1,
                         1,
                         1])

        self.shift = torch.Tensor(shift_list).to(self.device)
        self.scale = torch.Tensor(scale_list).to(self.device)
        self.num_inputs = len(shift_list)
        
    def prefix(self):
        return '%d%s%s%s' % (self.POLY_DEGREE,
                             't' if self.INCLUDE_SIN_COSINE else 'f',
                             't' if self.SPLINE else 'f',
                             't' if self.BOUNDED_FEATURES else 'f')

    def num_actions(self):
        return self.num_steer_positions * self.num_accel_positions

    def initial_W(self):
        return torch.randn(self.num_inputs ** self.POLY_DEGREE).to(self.device)

    def x_adjust(self, juncture, lane, speed, direction, steer, accel):
        ''' Takes the input params and converts it to an input feature array
        '''
        x_list = [1, lane, speed, direction, steer, accel]

        if self.SPLINE:
            for knot in range(self.SPLINE_LENGTH, self.num_junctures,
                              self.SPLINE_LENGTH):
                x_list.append(max(0, -(juncture - knot)))
        else:
            x_list.append(juncture)
        
        if self.INCLUDE_SIN_COSINE:
            # feature engineering
            sj = math.sin(juncture * 2 * math.pi / self.num_junctures)
            cj = math.cos(juncture * 2 * math.pi / self.num_junctures)
            sd = math.sin(direction * 2 * math.pi / self.num_directions)
            cd = math.cos(direction * 2 * math.pi / self.num_directions)
            x_list.extend([sj, cj, sd, cd])
            
        if self.BOUNDED_FEATURES:
            x_list.append(max(0, lane - 1))
            x_list.append(max(0, 3 - lane))
            x_list.append(max(0, steer - 1))
            x_list.append(max(0, 1 - steer))
            x_list.append(max(0, accel - 1))

        with torch.no_grad():
            x1 = torch.Tensor(x_list).to(self.device)
            x1 -= self.shift
            x1 /= self.scale
    
            x = x1
            if self.POLY_DEGREE >= 2:
                x = torch.ger(x, x1).flatten()
                if self.POLY_DEGREE == 3:
                    x = torch.ger(x, x1).flatten()
                elif self.POLY_DEGREE == 4:
                    x = torch.ger(x, x).flatten()

        return x
