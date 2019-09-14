'''
Created on Sep 14, 2018

@author: enerve
'''

import logging
import numpy as np
import random
import torch

from really.epoch_trainer import EpochTrainer
from really.agent import *
from really.function import *
from really import cmd_line
from really import log
from really import util

from racecar.car import Car
from racecar.track import LineTrack
from racecar.episode_factory import EpisodeFactory
from racecar.racecar_es_lookup import RacecarESLookup
from racecar.racecar_explorer import RacecarExplorer
from racecar.racecar_feature_eng import RacecarFeatureEng
from racecar.evaluator import Evaluator
import racecar.trainer_helper as th

def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'RC2 X'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar2")
    logger.setLevel(logging.DEBUG)

    # --------------
    
    points = [
            (45, 10),
            (80, 10),
            (120, 90),
            (160, 10),
            (230, 10),
            (195, 110),
            (230, 210),
            (160, 210),
            (120, 130),
            (80, 210),
            (10, 210),
            (45, 110),
            (10, 10)
        ]

    config = th.CONFIG(
        NUM_JUNCTURES = 200,
        NUM_MILESTONES = 200,
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
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    NUM_NEW_EPISODES = 100*1000
    NUM_EPOCHS = 1
    
    logger.debug("NUM_NEW_EPISODES=%d\t NUM_EPOCHS=%d", NUM_NEW_EPISODES, NUM_EPOCHS)
        
    episode_factory = EpisodeFactory(config, track, car)

    agent_fa = QLookup(config,
                        alpha=0.4,
                        feature_eng=RacecarFeatureEng(config))

    training_data_collector = FADataCollector(agent_fa)
    validation_data_collector = FADataCollector(agent_fa)

    es = RacecarESLookup(config,
                  explorate=19,
                  fa=agent_fa)
    explorer = RacecarExplorer(config, es)
    
    learner = th.create_agent(config, 
                    alg = 'sarsalambda',
                    lam = 0.9,
                    fa=agent_fa)


    # ------------------ Training -------------------

    test_agent = FAAgent(config, agent_fa)
    evaluator = Evaluator(episode_factory, test_agent)

    trainer = EpochTrainer(episode_factory, [explorer], learner, 
                           training_data_collector,
                           validation_data_collector,
                           evaluator,
                           explorer.prefix() + "_" + learner.prefix())
    
#     subdir = "212870_RC X_sarsa_lambda_400_0.2_1.0_0.9_"
#     driver.load_model(subdir)
#     trainer.load_stats(subdir)
    trainer.train(NUM_NEW_EPISODES, NUM_EPOCHS, 1)

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

    # --------- CV ---------
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
