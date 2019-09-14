'''
Created on Sep 14, 2018

@author: enerve
'''

import logging

from really.epoch_trainer import EpochTrainer
from really.agent import *
from really.function import *
from really import cmd_line
from really import log
from really import util

from racecar.car import Car
from racecar.track import CircleTrack
from racecar.episode_factory import EpisodeFactory
from racecar.circle_feature_eng import CircleFeatureEng
from racecar.circle_sa_feature_eng import CircleSAFeatureEng
from racecar.racecar_es_lookup import RacecarESLookup
from racecar.racecar_explorer import RacecarExplorer
from racecar.evaluator import Evaluator
import racecar.trainer_helper as th

import numpy as np
import random
import torch


def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'RC2 circle'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar2")
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
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    NUM_NEW_EPISODES = 7000
    NUM_EPOCHS = 1
    
    logger.debug("NUM_NEW_EPISODES=%d\t NUM_EPOCHS=%d", NUM_NEW_EPISODES, NUM_EPOCHS)

    episode_factory = EpisodeFactory(config, track, car)

    circle_fe = CircleFeatureEng(config)

    agent_fa = QLookup(config,
                       alpha=0.2,
                       feature_eng=circle_fe)
    
#     agent_fa =  PolynomialRegression(
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
#     agent_fa = MultiPolynomialRegression(
#                     0.0001, # alpha ... #4e-5 old alpha without batching
#                     0.5, # regularization constant
#                     256, # batch_size
#                     200, # max_iterations
#                     0.000, # dampen_by
#                     circle_fe)

    es = RacecarESLookup(config,
                  explorate=70,
                  fa=agent_fa)
    explorer = RacecarExplorer(config, es)

    learner = th.create_agent(config, 
                    alg = 'sarsalambda',
                    lam = 0.4,
                    fa=agent_fa)

    training_data_collector = FADataCollector(agent_fa)
    validation_data_collector = FADataCollector(agent_fa)

    # ------------------ Training -------------------

    test_agent = FAAgent(config, agent_fa)
    evaluator = Evaluator(episode_factory, test_agent)

    trainer = EpochTrainer(episode_factory, [explorer], learner, 
                           training_data_collector,
                           validation_data_collector,
                           evaluator,
                           explorer.prefix() + "_" + learner.prefix())
    
    #trainer.load_from_file("")

    # QLookup 4-explored 8000 episode QLambda-updated Qlookup 
    #trainer.load_from_file("439945_RC circle_DR_q_lambda_76_0.35_Qtable_a0.7_T_poly_a0.01_r0.002_b256_i50000_3ttt__")
    # QLookup 4-explored 8000 episode SarsaLambda-updated Qlookup 
    #trainer.load_from_file("462156_RC circle_DR_sarsa_lambda_e70_l0.40_Qtable_a0.2___")

    trainer.train(NUM_NEW_EPISODES, NUM_EPOCHS, 1)
    #trainer.save_to_file()
    
    explorer.store_episode_history("explorer")
    es.store_exploration_state()
    training_data_collector.store_last_dataset("final_t")
    validation_data_collector.store_last_dataset("final_v")
    agent_fa.save_model("v3")

    trainer.report_stats()
    trainer.save_stats()

    replay_agent = FAExplorer(config, ESBest(config, agent_fa))
    episode = episode_factory.new_episode([replay_agent])
    episode.start_recording()
    episode.run()
    
    logger.debug("Driver best episode total R = %0.2f time=%d", 
                 replay_agent.G,
                 episode.total_time_taken())
    episode.report_history()
    episode.play_movie(show=True, pref="bestmovie")#pref="bestmovie_%s" % pref)
    


if __name__ == '__main__':
    main()

