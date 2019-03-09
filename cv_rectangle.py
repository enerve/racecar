'''
Created on 7 Mar 2019

@author: enerve
'''

from car import Car
from track import LineTrack
from trainer import Trainer
from function import *
from rectangle_feature_eng import RectangleFeatureEng
import cmd_line
import log
import util
import numpy as np
import random
import torch
import trainer_helper as th
import logging

def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'CV RC rect'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar")
    logger.setLevel(logging.DEBUG)

def cv(fname):
    logger = logging.getLogger()
    
    # -------------- Configure track
    points = [
            (120, 40),
            (210, 40),
            (210, 180),
            (30, 180),
            (30, 40)
        ]
    config = th.CONFIG(
        NUM_JUNCTURES = 50,
        NUM_MILESTONES = 50,
        NUM_LANES = 5,
        NUM_SPEEDS = 3,
        NUM_DIRECTIONS = 20,
        NUM_STEER_POSITIONS = 3,
        NUM_ACCEL_POSITIONS = 3
    )

    WIDTH = 20
    track = LineTrack(points, WIDTH, config)
    
    car = Car(config)
    
    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   %s", config)

    # --------- CV ---------
    num_samples = 2
    
    i_algs = np.array([x % 4 for x in range(num_samples)])
    fas = np.array([th.FA['qtable'] for _ in range(num_samples)])
    lambdas = np.random.uniform(0.0, 1, num_samples) 
    alphas = np.random.uniform(0.0, 1, num_samples)
    expls = 10 ** np.random.uniform(1.0, 2.0, num_samples)
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
     
    
#     explorates = [50, 100, 200, 400, 800]
#     stats_bp_times = []
#     stats_e_bp = []
#     stats_labels = []
#     num_episodes = 60 * 1000
#     for explorate in explorates:
#         logger.debug("--- Explorate=%d ---" % explorate)
#         #for i in range(3):
#         i=0
#         seed = (100 + 53*i)
#         pref = "%d_%d" % (explorate, seed)
#         driver = QDriver(1, # alpha
#                         1, # gamma
#                         explorate, # explorate
#                         NUM_JUNCTURES,
#                         NUM_LANES,
#                         NUM_SPEEDS,
#                         NUM_DIRECTIONS,
#                         NUM_STEER_POSITIONS,
#                         NUM_ACCEL_POSITIONS)
#         stat_bestpath_times, stat_e_bp = \
#             trainer.train(driver, track, car, num_episodes, seed=seed, pref=pref)
#         stats_bp_times.append(stat_bestpath_times)
#         stats_e_bp.append(stat_e_bp)
#         stats_labels.append("N0=%d seed=%d" % (explorate, seed))
#         logger.debug("bestpath: %s", stat_bestpath_times)
#         logger.debug("stat_e: %s", stat_e_bp)
#         trainer.play_best(driver, track, car, should_play_movie=False,
#                           pref=pref)
#     util.plot_all(stats_bp_times, stats_e_bp, stats_labels,
#                   title="Time taken by best path as of epoch", pref="BestTimeTaken")

def plot(fname):
    iAlg, iFa, il, ia, ie, iS = range(6) 
    alg = ''
    
    A_ = util.load(fname)
    A_ = A_[A_[:, il] > 0.6]
    #A_ = A_[A_[:, ia] < 0.8]
    #A_ = A_[A_[:, ie] < 200]
    #alg = 'sarsalambda'; A_ = A_[A_[:, iAlg] == th.ALG[alg]]

    lambdas = A_[:, il]
    alphas = A_[:, ia]
    expls = A_[:, ie]
    scores = A_[:, iS]

    # TODO: fix plot file name
    scatter_all(lambdas, alphas, expls, scores, pref=alg+"_")

if __name__ == '__main__':
    main()
    
    #cv("rect_cvdump")
    plot("rect_cvdump")
