'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from car import Car
from track import CircleTrack
from epoch_trainer import EpochTrainer
from driver import *
from function import MultiPolynomialRegression
from function import PolynomialRegression
from function import QLookup
import cmd_line
import log
import numpy as np
import random
import util


def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'RC circle'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar")
    logger.setLevel(logging.DEBUG)
    
    # --------------
    
    NUM_JUNCTURES = 28
    NUM_MILESTONES = 27
    NUM_LANES = 5
    MAX_SPEED = NUM_SPEEDS = 3
    NUM_DIRECTIONS = 20
    
    NUM_STEER_POSITIONS = 3
    NUM_ACCEL_POSITIONS = 3

    RADIUS = 98
    WIDTH = 20
    track = CircleTrack((0, 0), RADIUS, WIDTH, NUM_JUNCTURES,
                        NUM_MILESTONES, NUM_LANES)
    car = Car(NUM_DIRECTIONS, NUM_SPEEDS)

    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   NUM_JUNCTURES:\t%s", NUM_JUNCTURES)
    logger.debug("   NUM_MILESTONES:\t%s", NUM_MILESTONES)
    logger.debug("   NUM_LANES:\t%s", NUM_LANES)
    logger.debug("   MAX_SPEED:\t%s", MAX_SPEED)
    logger.debug("   NUM_DIRECTIONS:\t%s", NUM_DIRECTIONS)
    logger.debug("   NUM_STEER_POSITIONS:\t%s", NUM_STEER_POSITIONS)
    logger.debug("   NUM_ACCEL_POSITIONS:\t%s", NUM_ACCEL_POSITIONS)
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    # ------------------ Guide driver FA -----------------
#     fa_Q = QLookup(0.7,  # alpha
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
    
#     fa_Poly =  PolynomialRegression(
#                     0.002, # alpha ... #4e-5 old alpha without batching
#                     0.5, # regularization constant
#                     256, # batch_size
#                     5000, #250, # max_iterations
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)        
    fa_Multi = MultiPolynomialRegression(
                    0.0001, # alpha ... #4e-5 old alpha without batching
                    0.05, # regularization constant
                    256, # batch_size
                    200, # max_iterations
                    NUM_JUNCTURES,
                    NUM_LANES,
                    NUM_SPEEDS,
                    NUM_DIRECTIONS,
                    NUM_STEER_POSITIONS,
                    NUM_ACCEL_POSITIONS,
                    dampen_by=0.000)

    # ------------------ Guide driver RL algorithm ---------------

#     driver = SarsaFADriver(
#                     1, # gamma
#                     200, # explorate
#                     fa_Q,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     driver = SarsaLambdaFADriver(
#                     0.85, #lambda
#                     1, # gamma
#                     76, # explorate
#                     fa_Q,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     driver = QFADriver(
#                     1, # gamma
#                     200, # explorate
#                     fa_Q,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
    driver = QLambdaFADriver(
                    0.35, #lambda
                    1, # gamma
                    76, # explorate
                    fa_Multi,
                    NUM_JUNCTURES,
                    NUM_LANES,
                    NUM_SPEEDS,
                    NUM_DIRECTIONS,
                    NUM_STEER_POSITIONS,
                    NUM_ACCEL_POSITIONS)

    # ------------------ Student driver FA -----------------
#     fa_Q_S = QLookup(0.7,  # alpha
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     fa_Poly_S =  PolynomialRegression(
#                     0.05, # alpha ... #4e-5 old alpha without batching
#                     0, # regularization constant
#                     256, # batch_size
#                     1000, #250, # max_iterations
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     fa_Multi_S = MultiPolynomialRegression(
#                     0.0002, # alpha ... #4e-5 old alpha without batching
#                     0.002, # regularization constant
#                     0, # batch_size
#                     200, # max_iterations
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)

    # ------------------ Student driver RL algorithm -------------
    student = None

#     student = QFAStudent(
#                     1, # gamma
#                     fa_Q_S,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     student = SarsaLambdaFAStudent(
#                     1, #lambda
#                     1, # gamma
#                     fa_Poly_S,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     student = QLambdaFAStudent(
#                     0.35, #lambda
#                     1, # gamma
#                     fa_Multi_S,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
    
    # ------------------ Training -------------------

    #trainer = Trainer(driver, track, car)
    trainer = EpochTrainer(driver, track, car, student)
    
    #trainer.load_from_file("")

    # QLookup 4-explored 8000 episode QLambda-updated Qlookup 
    #trainer.load_from_file("439945_RC circle_DR_q_lambda_76_0.35_Qtable_a0.7_T_poly_a0.01_r0.002_b256_i50000_3ttt__")

    trainer.train(10, 500, 25)

#     driver.TEST_FA = False
#     trainer.train(1, 8000, 3)
    trainer.save_to_file()
#     driver.TEST_FA = True
#     trainer.train(1, 8000, 1)
    
    logger.debug("Training complete")
    trainer.report_stats()
    
    #     # Load poly training data and train
    #     subdir = "71177_RC circle_sarsa_lambda_fa_student_1.0_1.0_" # Qlambda driver 
    #                                                                 # 1 epoch 8000
    #     fa_Poly_S.load_training_data("student_train", subdir)
    #     #fa_Poly_S.describe_training_data()
    #     fa_Poly_S.train()
    #     fa_Poly_S.report_stats()

    t_R, b_E, _ = driver.run_best_episode(track, car, False)
    logger.debug("Driver best episode total R = %0.2f time=%d", t_R,
                 b_E.total_time_taken())
    b_E.play_movie(pref="bestmovie")

    if driver.TEST_FA:
        t_R, b_E, _ = driver.run_best_episode(track, car, True)
        logger.debug("TestDriver best episode total R = %0.2f time=%d", t_R,
                     b_E.total_time_taken())
        b_E.play_movie(pref="bestmovie_testfa")

    if student:
        t_R, b_E, _ = student.run_best_episode(track, car)
        logger.debug("Student best episode total R = %0.2f time=%d", t_R,
                 b_E.total_time_taken())
        b_E.play_movie(pref="bestmovie_student")


    # --------- CV ---------
#     import random
#     lambdas = []
#     alphas = []
#     scores = []
#     expls = []
# 
# 
#     for rep in range(20):
#         lam = random.random() 
#         alp = random.uniform(0.3, 1.0)
#         expl = 10 ** random.uniform(1.0, 3)
#         logger.debug("--- rep %d --- lam: %0.2f, alp: %0.2f, expl: %0.2f", 
#                      rep, lam, alp, expl)
#         
#         driver = QLambdaDriver(lam, # lambda
#                         alp, # alpha
#                         1, # gamma
#                         expl, # explorate
#                         NUM_JUNCTURES,
#                         NUM_LANES,
#                         NUM_SPEEDS,
#                         NUM_DIRECTIONS,
#                         NUM_STEER_POSITIONS,
#                         NUM_ACCEL_POSITIONS)
#         trainer = Trainer(driver, track, car)
#         bp_times, e_bp, bp_R, bp_j = trainer.train(20*1000, save_to_file=False,
#                                              seed = 513 + rep)
#         #bp_R=[random.randrange(20, 1000)]
#                 
#         lambdas.append(lam)
#         alphas.append(alp)
#         expls.append(expl)
#                
#         score = 0
#         for i in range(5):
#             score += bp_R[-1-i] / bp_j[-1-i]
#         score /= 5
#         scores.append(score)
#         logger.debug("  Score: %s", score)
#             
#  
#  
#         logger.debug("lambdas.extend( %s)", lambdas)
#         logger.debug("alphas.extend(  %s)", alphas)
#         logger.debug("expls.extend(   %s)", expls)
#         logger.debug("scores.extend(  %s)", scores)
#        
#     util.scatter(np.array(lambdas), np.array(alphas), np.array(scores),
#                  "lambda", "alpha", pref="cv_la")
#     util.scatter(np.array(lambdas), np.array(expls), np.array(scores),
#                  "lambda", "expl", pref="cv_le")
#     util.scatter(np.array(expls), np.array(alphas), np.array(scores),
#                  "expl", "alpha", pref="cv_ea")


#     explorates = [10, 100, 1000, 10000]
#     stats_bp_times = []
#     stats_e_bp = []
#     stats_labels = []
#     num_episodes = 300 * 1000
#     for explorate in explorates:
#         logger.debug("--- Explorate=%d ---" % explorate)
#         for i in range(3):
#             seed = (100 + 53*i)
#             pref = "%d_%d" % (explorate, seed)
#             driver = Driver(alpha=1, gamma=1, explorate=explorate)
#             stat_bestpath_times, stat_e_bp = \
#                 train(driver, track, car, num_episodes, seed=seed, pref=pref)
#             stats_bp_times.append(stat_bestpath_times)
#             stats_e_bp.append(stat_e_bp)
#             stats_labels.append("N0=%d seed=%d" % (explorate, seed))
#             logger.debug("bestpath: %s", stat_bestpath_times)
#             logger.debug("stat_e: %s", stat_e_bp)
#             play_best(driver, track, car, should_play_movie=False,
#                       pref=pref)
#     util.plot_all(stats_bp_times, stats_e_bp, stats_labels,
#                   title="Time taken by best path as of epoch", pref="BestTimeTaken")

if __name__ == '__main__':
    main()
