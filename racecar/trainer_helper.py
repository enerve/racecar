'''
Created on 8 Mar 2019

@author: enerve
'''

from really.learner import *
from really.function import *
from collections import namedtuple

ALG = {
        'q' : 0,
        'sarsa' : 1,
        'qlambda' : 2,
        'sarsalambda' : 3
    }

FA = {
        'qtable' : 0,
        'poly' : 1,
        'multi' : 2,
        'nn' : 3
    }

CONFIG = namedtuple("Config",
    "NUM_JUNCTURES, NUM_MILESTONES, NUM_LANES, NUM_SPEEDS, NUM_DIRECTIONS, NUM_STEER_POSITIONS, NUM_ACCEL_POSITIONS")

# TODO: This is too hidden
GAMMA = 1.0

def create_agent(config, alg, lam, fa):
    return create_agent_i(config, ALG[alg], lam, fa)

def create_agent_i(config, i_alg, lam, fa):
    if i_alg == 0:    
        agent = QLearner(config,
                        GAMMA, # gamma
                        fa)
    elif i_alg == 1:    
        agent = SarsaLearner(config,
                        GAMMA, # gamma
                        fa)
    elif i_alg == 2:
        agent = QLambdaLearner(config,
                        lam, #lambda
                        GAMMA, # gamma
                        fa)
    elif i_alg == 3:
        agent = SarsaLambdaLearner(config,
                        lam, #lambda
                        GAMMA, # gamma
                        fa)
    return agent
