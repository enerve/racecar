'''
Created on Sep 6, 2018

@author: enerve
'''

import logging
import numpy as np
import random

from environment import Environment
import util

class Driver(object):
    '''
    An agent that controls the race car
    '''

    NUM_JUNCTURES = 28
    NUM_LANES = 5
    MAX_SPEED = 3
    NUM_DIRECTIONS = 20
    
    NUM_STEER_POSITIONS = 3
    NUM_ACCEL_POSITIONS = 3

    def __init__(self, alpha, gamma, explorate, load_filename=None):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_alg = "qlearn"
        
        self.alpha = alpha  # learning rate for updating Q
        self.gamma = gamma  # weight given to predicted future
        self.explorate = explorate # Inclination to explore, e.g. 0, 10, 1000
        
        # Q is the learned value of a state/action
        if load_filename is not None:
            stored_Q = util.load(load_filename)
            self.Q = stored_Q.reshape((Driver.NUM_JUNCTURES,
                                       Driver.NUM_LANES,
                                       Driver.MAX_SPEED,
                                       Driver.NUM_DIRECTIONS,
                                       Driver.NUM_STEER_POSITIONS,
                                       Driver.NUM_ACCEL_POSITIONS))
        else:
            self.Q = np.zeros((Driver.NUM_JUNCTURES,
                               Driver.NUM_LANES,
                               Driver.MAX_SPEED,
                               Driver.NUM_DIRECTIONS,
                               Driver.NUM_STEER_POSITIONS,
                               Driver.NUM_ACCEL_POSITIONS))
                           
        # C is the count of visits to state/action
        self.C = np.zeros((Driver.NUM_JUNCTURES,
                           Driver.NUM_LANES,
                           Driver.MAX_SPEED,
                           Driver.NUM_DIRECTIONS,
                           Driver.NUM_STEER_POSITIONS,
                           Driver.NUM_ACCEL_POSITIONS), dtype=np.int32)
        # C is the count of visits to state
        self.N = np.zeros((Driver.NUM_JUNCTURES,
                           Driver.NUM_LANES,
                           Driver.MAX_SPEED,
                           Driver.NUM_DIRECTIONS), dtype=np.int32)
        # Rs is the average reward at juncture (for statistics)
        self.Rs = np.zeros((Driver.NUM_JUNCTURES), dtype=np.float)

    def pick_action(self, S, run_best):  
        if run_best:
            epsilon = 0
        else:
            n = self.N[S]
            N0 = self.explorate
            epsilon = N0 / (N0 + n)
      
        if random.random() >= epsilon:
            # Pick best
            As = self.Q[S]            
            steer, accel = np.unravel_index(np.argmax(As, axis=None), As.shape)
            
            #debugging
            #if run_best:
            #    my_s, my_a = MY_IDEAL_A[S[0]]
            #    self.logger.debug("I prefer: %d whose Q is %0.2f or %0.2f ",
            #                      my_s, As[my_s, 1], As[my_s, 2])
            #    self.logger.debug("  Chosen: %d, %d whose Q is %0.2f ... at %s", 
            #                      steer, accel, As[steer, accel], S)
        else:
            r = random.randrange(
                Driver.NUM_STEER_POSITIONS * Driver.NUM_ACCEL_POSITIONS)
            steer = r % Driver.NUM_STEER_POSITIONS
            accel = r // Driver.NUM_STEER_POSITIONS
        
        return (steer, accel)
    
    def max_at_state(self, S):
        if S is None:
            return 0
        # TODO: Add some kind of nearby-approximation?
        return np.max(self.Q[S])
        
    def q_index(self, state, action):
        m, l, v, d = state
        s, a = action
        return (m, l, v, d, s, a)
    

    def run_episode(self, track, car, run_best=False):
        environment = Environment(track,
                                  car,
                                  Driver.NUM_JUNCTURES,
                                  should_record=run_best)
        S = environment.state_encoding() # 0 2 0 0
        total_R = 0
        
        while S is not None:
            A = self.pick_action(S, run_best)
            
            I = self.q_index(S, A)

            R, S_ = environment.step(A)

            target = R + self.gamma * self.max_at_state(S_)
            self.Q[I] += self.alpha * (target - self.Q[I])
            self.C[I] += 1
            self.N[S] += 1
            if S_ is not None:
                self.Rs[S_[0]] += 0.1 * (R - self.Rs[S_[0]])
            
            S = S_
            total_R += R
            
        return total_R, environment
            
    def Q_to_plot(self, m, axes, pick_max=False):
        M = m
        Q = self.Q[0:M, :, :, :, :, :]
        if pick_max:
            Q = np.max(Q, axis=axes)
        else:
            Q = np.mean(Q, axis=axes)
        Q = Q.reshape(M, -1).T
        #self.logger.debug("%s", Q.astype(np.int32))
        return Q

    def C_to_plot(self, m, axes):
        M = m
        C = self.C[0:M, :, :, :, :, :]
        C = np.sum( C, axis=axes)
        C = C.reshape(M, -1).T
        #self.logger.debug("%s", C)
        return C

    def plotQ(self, m, axes, t_suffix):
        M = m
        Q = self.Q_to_plot(m, axes)
        util.heatmap(Q, None, "Q total per juncture %s" % t_suffix, pref="Q")
        
        C = self.C_to_plot(m, axes)    
        util.heatmap(C, None, "Total updates per juncture %s" % t_suffix, pref="C")

    def dumpQ(self, pref=""):
        M = Driver.NUM_JUNCTURES
        util.dump(self.Q.reshape(M, -1), "Q_%s_%s" % (M, pref))
        #util.dump(self.C.reshape(M, -1), "C_%s" % M)
        #util.dump(self.N.reshape(M, -1), "N_%s" % M)

            
