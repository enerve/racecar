'''
Created on 8 Mar 2019

@author: enerve
'''

from driver import *
from function import *
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

def create_fa(config, i_fa, alpha):
    if i_fa == 0:
        fa = QLookup(config, alpha)
    return fa

def create_driver(config, alg, expl, lam, fa, mimic_fa):
    return create_driver_i(config, ALG[alg], expl, lam, fa, mimic_fa)

def create_driver_i(config, i_alg, expl, lam, fa, mimic_fa):
    if i_alg == 0:    
        driver = QFADriver(config,
                        1, # gamma
                        expl, # explorate
                        fa,
                        mimic_fa)
    elif i_alg == 1:    
        driver = SarsaFADriver(config,
                        1, # gamma
                        expl, # explorate
                        fa,
                        mimic_fa)
    elif i_alg == 2:
        driver = QLambdaFADriver(config,
                        lam, #lambda
                        1, # gamma
                        expl, # explorate
                        fa,
                        mimic_fa)
    elif i_alg == 3:
        driver = SarsaLambdaFADriver(config,
                        lam, #lambda
                        1, # gamma
                        expl, # explorate
                        fa,
                        mimic_fa)
    return driver

def create_student(config, alg, lam, fa):
    return create_student_i(config, ALG[alg], lam, fa)

def create_student_i(config, i_alg, lam, fa):
    if i_alg == 0:
        driver = QFAStudent(config,
                        1, # gamma
                        fa,
                        None)
    elif i_alg == 1:    
        driver = SarsaFAStudent(config,
                        1, # gamma
                        fa,
                        None)
    elif i_alg == 2:
        driver = QLambdaFAStudent(config,
                        lam, #lambda
                        1, # gamma
                        fa,
                        None)
    elif i_alg == 3:
        driver = SarsaLambdaFAStudent(config,
                        lam, #lambda
                        1, # gamma
                        fa,
                        None)
    return driver
