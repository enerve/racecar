'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from car import Car
from track import CircleTrack
from epoch_trainer import EpochTrainer
from trainer import Trainer
from driver import *
from function import *
from circle_feature_eng import CircleFeatureEng
import trainer_helper as th
import cmd_line
import log
import util

import numpy as np
import random
import torch


def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'RC circle'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar")
    logger.setLevel(logging.DEBUG)
    
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
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ------------------ Guide driver FA -----------------
    driver_fa = QLookup(config,
                        alpha=0.8)
    
#     driver_fa =  PolynomialRegression(
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
#     fe = CircleFeatureEng(
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     driver_fa = MultiPolynomialRegression(
#                     0.0001, # alpha ... #4e-5 old alpha without batching
#                     0.5, # regularization constant
#                     256, # batch_size
#                     200, # max_iterations
#                     0.000, # dampen_by
#                     fe)

    # ------------------ Mimic FA -----------------
    mimic_fa = None

#     mimic_fa = PolynomialRegression(
#                     0.01, # alpha ... #4e-5 old alpha without batching
#                     0.002, # regularization constant
#                     256, # batch_size
#                     50000, # max_iterations
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)        

#     fe = CircleFeatureEng(
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     mimic_fa = MultiPolynomialRegression(
#                     0.0002, # alpha ... #4e-5 old alpha without batching
#                     0.002, # regularization constant
#                     256, # batch_size
#                     500, # max_iterations
#                     0.000, # dampen_by
#                     fe)    

    # ------------------ Guide driver RL algorithm ---------------

    driver = th.create_driver(config, 
                    alg = 'qlambda', 
                    expl = 110,
                    lam = 0.25,
                    fa=driver_fa,
                    mimic_fa=mimic_fa)


    # ------------------ Student driver FA -----------------
#     student_fa = QLookup(0.7,  # alpha
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     student_fa =  PolynomialRegression(
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
#     student_fa = MultiPolynomialRegression(
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
#                     student_fa,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     student = SarsaLambdaFAStudent(
#                     1, #lambda
#                     1, # gamma
#                     student_fa,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
#     student = QLambdaFAStudent(
#                     0.35, #lambda
#                     1, # gamma
#                     student_fa,
#                     NUM_JUNCTURES,
#                     NUM_LANES,
#                     NUM_SPEEDS,
#                     NUM_DIRECTIONS,
#                     NUM_STEER_POSITIONS,
#                     NUM_ACCEL_POSITIONS)
    
    # ------------------ Training -------------------

    trainer = Trainer(driver, track, car)
    trainer.train(20000)
    
    #trainer = EpochTrainer(driver, track, car, student)
    
    #trainer.load_from_file("")

    # QLookup 4-explored 8000 episode QLambda-updated Qlookup 
    #trainer.load_from_file("439945_RC circle_DR_q_lambda_76_0.35_Qtable_a0.7_T_poly_a0.01_r0.002_b256_i50000_3ttt__")

    #trainer.train(5, 500, 1)

#     driver.mimic_fa = False
#     trainer.train(1, 8000, 3)
    trainer.save_to_file()
#     driver.mimic_fa = mimic_fa
#     trainer.train(1, 8000, 1)
    
    trainer.report_stats()
    
    #     # Load poly training data and train
    #     subdir = "71177_RC circle_sarsa_lambda_fa_student_1.0_1.0_" # Qlambda driver 
    #                                                                 # 1 epoch 8000
    #     fa_Poly_S.load_training_data("student_train", subdir)
    #     #fa_Poly_S.describe_training_data()
    #     fa_Poly_S.train()
    #     fa_Poly_S.report_stats()

    if True:
        t_R, b_E, _ = driver.run_best_episode(track, car, False)
        logger.debug("Driver best episode total R = %0.2f time=%d", t_R,
                     b_E.total_time_taken())
        #b_E.play_movie(pref="bestmovie")
    
        if driver.mimic_fa:
            t_R, b_E, _ = driver.run_best_episode(track, car, True)
            logger.debug("Mimic best episode total R = %0.2f time=%d", t_R,
                         b_E.total_time_taken())
            b_E.play_movie(pref="bestmovie_mimic")
    
        if student:
            t_R, b_E, _ = student.run_best_episode(track, car)
            logger.debug("Student best episode total R = %0.2f time=%d", t_R,
                     b_E.total_time_taken())
            b_E.play_movie(pref="bestmovie_student")



if __name__ == '__main__':
    main()

