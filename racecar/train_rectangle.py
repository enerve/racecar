'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from car import Car
from driver import *
from track import LineTrack
from trainer import Trainer
from epoch_trainer import EpochTrainer
from function import MultiPolynomialRegression
from function import QLookup
from function import NN_FA
from rectangle_feature_eng import RectangleFeatureEng
import cmd_line
import log
import trainer_helper as th
import util

import numpy as np
import random
from collections import namedtuple

def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'RC rect'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar")
    logger.setLevel(logging.DEBUG)

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
    
    #util.draw_image(track.draw())
    
    car = Car(config)
    
    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   %s", config)

    seed = 123
    random.seed(seed)
    np.random.seed(seed)

    # ------------------ Guide driver FA -----------------

    driver_fa = QLookup(config,
                        alpha=0.5)

#     driver_fe = RectangleFeatureEng(config)
#  
#     driver_fa = MultiPolynomialRegression(
#                     0.002, # alpha ... #4e-5 old alpha without batching
#                     0.005, # regularization constant
#                     256, # batch_size
#                     1000, # max_iterations
#                     0.000, # dampen_by
#                     driver_fe)

    # ------------------ Mimic FA -----------------
    mimic_fa = None
 
#     mimic_fe = RectangleFeatureEng(
#                     config,
#                     include_basis = True,
#                     include_sin_cosine = True,
#                     include_splines = True,
#                     spline_length = 4,
#                     include_corner_splines = True,
#                     corners = [0, 4, 15, 33, 44, 49],
#                     include_bounded_features = True,
#                     poly_degree = 2)
#     mimic_fa = MultiPolynomialRegression(
#                     10.0, # alpha ... #4e-5 old alpha without batching
#                     0.0001, # regularization constant
#                     512, # batch_size
#                     3000, # max_iterations
#                     0.001, # dampen_by
#                     mimic_fe)
    
#     mimic_fe2 = RectangleFeatureEng(config)
#     mimic_fa = NN_FA(
#                     0.000001, # alpha ... #4e-5 old alpha without batching
#                     1, # regularization constant
#                     512, # batch_size
#                     2000, # max_iterations
#                     mimic_fe2)
     
    # ------------------ Guide driver RL algorithm ---------------

    driver = th.create_driver(config, 
                    alg = 'sarsalambda',
                    expl = 30, #200
                    lam = 0.5, #0.8
                    fa=driver_fa,
                    mimic_fa=mimic_fa)


    # ------------------ Student driver FA -----------------

    student_fa = None

#     student_fe = RectangleFeatureEng(config) 
#     student_fa = MultiPolynomialRegression(
#                     0.00001, # alpha ... #4e-5 old alpha without batching
#                     0.005, # regularization constant
#                     256, # batch_size
#                     100, # max_iterations
#                     0.000, # dampen_by
#                     student_fe)

    student_fe2 = RectangleFeatureEng(config)
    student_fa = NN_FA(
                    0.000001, # alpha ... #4e-5 old alpha without batching
                    1, # regularization constant
                    512, # batch_size
                    2000, # max_iterations
                    student_fe2)
    
    # ------------------ Student driver RL algorithm -------------
    student = None
 
    student = th.create_student(config, 
                    alg = 'sarsalambda',
                    lam = 0.8,
                    fa=student_fa)
    
    # ------------------ Training -------------------
    
#     util.start_interactive()

    #subdir = "97640_RC rect_DR_sarsa_lambda_e50_l0.50_Qtable_a0.4___" # good path
    #driver.load_model(subdir)
    #trainer = Trainer(driver, track, car)
    trainer = EpochTrainer(driver, track, car, student)
    #trainer.train(1000)
    
    #subdir = "215973_RC rect_DR_q_lambda_200_0.80_Qtable_a0.2_M_multipoly_a0.002_r0.005_b256_i200_d0.0000_F3tftt__"
    # 100 x 500 x 10 trained on top of above
    # subdir = "224148_RC rect_DR_q_lambda_200_0.80_Qtable_a0.2__ST_q_lambda_l0.80_multipoly_a1e-05_r0.005_b256_i100_d0.0000_F3tftt__"
    # 
    #trainer.load_stats(subdir)
    
    trainer.train(1, 1000, 1)
#     trainer.train(10, 2000, 2)
    #trainer.save_to_file()
#     #mimic_fa.store_training_data("mimic")
#     util.stop_interactive()

    trainer.report_stats()
    
#     # 20000 episodes worth of training data, built upon 100 x 500 x 10 pretrain
#     subdir = "311066_RC rect_DR_q_lambda_200_0.80_Qtable_a0.2_M_multipoly_a0.0002_r0.005_b256_i100_d0.0000_F3tftt__"
#     mimic_fa.load_training_data("mimic", subdir)
#     #mimic_fa.describe_training_data()
#     mimic_fa.train()
#     mimic_fa.test()
#     mimic_fa.report_stats("mimic")


    if driver:
        t_R, b_E, _ = driver.run_best_episode(track, car, False)
        logger.debug("Driver best episode total R = %0.2f time=%d", t_R,
                     b_E.total_time_taken())
    #     b_E.play_movie(pref="bestmovie")
    
        if driver.mimic_fa:
            t_R, b_E, _ = driver.run_best_episode(track, car, True)
            logger.debug("Mimic best episode total R = %0.2f time=%d", t_R,
                         b_E.total_time_taken())
            b_E.play_movie(pref="bestmovie_mimic")
    
        if student:
            t_R, b_E, _ = student.run_best_episode(track, car)
            student_fa.report_stats("student")
            logger.debug("Student best episode total R = %0.2f time=%d", t_R,
                     b_E.total_time_taken())
            b_E.play_movie(pref="bestmovie_student")

if __name__ == '__main__':
    main()
