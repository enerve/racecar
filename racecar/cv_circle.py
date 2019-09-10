'''
Created on 6 Mar 2019

@author: enerve
'''


from car import Car
import cmd_line
import log
from track import CircleTrack
from trainer import Trainer
from function import *
import trainer_helper as th
from circle_feature_eng import CircleFeatureEng
import util

import logging
import numpy as np
import random
import torch


def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'CV RC circle'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar")
    logger.setLevel(logging.DEBUG)
    
        
def cv(fname):
    logger = logging.getLogger()

    # -------------- Configure track
    config = th.CONFIG(
        NUM_JUNCTURES = 28,
        NUM_MILESTONES = 27,
        NUM_LANES = 5,
        NUM_SPEEDS = 3,
        NUM_DIRECTIONS = 20,
        NUM_STEER_POSITIONS = 3,
        NUM_ACCEL_POSITIONS = 3
    )

    RADIUS = 98
    WIDTH = 20
    track = CircleTrack((0, 0), RADIUS, WIDTH, config)
    car = Car(config)

    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   %s", config)

    # --------- CV ---------
    num_samples = 600

#     algs = np.array([ALG['sarsa'] for _ in range(num_samples)])
#     fas = np.array([FA['qtable'] for _ in range(num_samples)])
    i_algs = np.array([x % 4 for x in range(num_samples)])
    fas = np.array([th.FA['qtable'] for _ in range(num_samples)])
    lambdas = np.random.uniform(0.0, 1, num_samples)
    alphas = np.random.uniform(0.0, 1, num_samples)
    expls = 10 ** np.random.uniform(1.0, 3, num_samples)
    scores = []
    erjs = []

    for rep, (i_alg, fa, lam, alp, expl) in enumerate(
        zip(i_algs, fas, lambdas, alphas, expls)): 
        logger.debug("--- rep %d --- %d:%d lam: %0.2f, alp: %0.2f, expl: %0.2f", 
                     rep, i_alg, fa, lam, alp, expl)
         
        #TODO: use 'fa' to pick f.a.
        driver_fa = QLookup(config, alpha=alp)
        driver = th.create_driver_i(config, i_alg, expl, lam, driver_fa, None)
    
        trainer = Trainer(driver, track, car)
        seed = 213 + rep
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        bp_times, e_bp, bp_R, bp_j = trainer.train(20*1000)
        #bp_R=[random.randrange(20, 1000)]

        score = 0
        mult = 1
        for i in range(len(bp_R)):
            score += bp_R[-1-i] / (bp_j[-1-i] + 1) * mult
            mult *= 0.95

        scores.append(score)
        logger.debug("  Score: %s", score)
        erj = []
        erj.extend(e_bp)
        erj.extend(bp_R)
        erj.extend(bp_j)
        erjs.append(erj)
          
    scores = np.array(scores)
    erjs = np.array(erjs)
    
    stackers = [i_algs, fas, lambdas, alphas, expls, scores]
    stackers.extend([erj.T for erj in erjs.T])

    A = np.stack(stackers).T    
    util.append(A, fname)
    
def scatter_all(lambdas, alphas, expls, scores, pref=""):
    logger = logging.getLogger()
    
    logger.debug("lambdas: %s", lambdas)
    logger.debug("alphas:  %s", alphas)
    logger.debug("expls:   %s", expls)
    logger.debug("scores:  %s", scores)
    util.scatter(lambdas, alphas, scores,
                 "lambda", "alpha", pref=pref+"cv_la")
    util.scatter(lambdas, expls, scores,
                 "lambda", "explorate", pref=pref+"cv_le")
    util.scatter(expls, alphas, scores,
                 "explorate", "alpha", pref=pref+"cv_ea")

def plot(fname):
    iAlg, iFa, il, ia, ie, iS = range(6)
    alg = '' 
    
    # Filtering
    A_ = util.load(fname)
    #A_ = A_[A_[:, il] > 0.6]
    #A_ = A_[A_[:, ia] < 0.5]
    #A_ = A_[A_[:, ie] < 400]
    alg = 'qlambda'; A_ = A_[A_[:, iAlg] == th.ALG[alg]]

    # Scoring
    erj = A_[:, 6:]
    n = erj.shape[1] // 3
    e_bp = erj[:, 0:n]
    bp_R = erj[:, n:(2*n)]
    bp_j = erj[:, 2*n:]
#     S = np.zeros(A.shape[0])
#     mult = 1
#     for i in range(n):
#         S += bp_R[:, -1-i] / (bp_j[:, -1-i] + 1) * mult
#         mult *= 0.95

    crashers = (bp_j < 28)
    bp_R[crashers] += 200  # undo the crash penalty

    S = np.zeros(A_.shape[0])
    mult = 1
    for i in range(n):
        S += bp_R[:, -1-i] / (bp_j[:, -1-i] + 1) * mult
        mult *= 0.95

    lambdas = A_[:, il]
    alphas = A_[:, ia]
    expls = A_[:, ie]
#     scores = A_[:, iS]
    scores = S
    scatter_all(lambdas, alphas, expls, scores, pref=alg+"_")

if __name__ == '__main__':
    main()
    
    #cv("cvdump_erj_test")
    plot("cvdump_erj")
