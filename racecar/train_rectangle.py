'''
Created on Sep 1, 2019

@author: enerve
'''

import logging
import numpy as np
import random
import torch
import torch.nn as nn

from really.epoch_trainer import EpochTrainer
from really.agent import *
from really.agent import ESPolicy
from really.function import *
from really.function.nn_model import NNModel
from really.policy import PolicyLearner, PolicyGrader
from really import cmd_line
from really import log
from really import util

from racecar import *
import racecar.trainer_helper as th

#from .net import Net

def main():
    args = cmd_line.parse_args()

    util.init(args)
    util.pre_problem = 'RC2 rect'

    logger = logging.getLogger()
    log.configure_logger(logger, "RaceCar2")
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

    car = Car(config)

    logger.debug("*Problem:\t%s", util.pre_problem)
    logger.debug("   %s", config)
    
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    NUM_NEW_EPISODES = 100
    NUM_EPOCHS = 2
    MAX_FA_ITERATIONS = 200
    MAX_PA_ITERATIONS = 20

    logger.debug("NUM_NEW_EPISODES=%d\t NUM_EPOCHS=%d", NUM_NEW_EPISODES, NUM_EPOCHS)
        
    episode_factory = EpisodeFactory(config, track, car)

    rect_fe = RectangleFeatureEng(config,
                                  include_splines=True,
                                  spline_length=1)
    rect_sa_fe = RectangleSAFeatureEng(config,
                                  include_splines=True,
                                  spline_length=1)
    
    fe = rect_sa_fe

    nn_model = NNModel(
        Grader('mse'),
        'adam',
        0.0001, # alpha
        0.1, # regularization constant
        512, # batch_size
        MAX_FA_ITERATIONS)

#     agent_fa = S_FA(
#         config,
#         nn_model,
#         fe)
    agent_fa = SA_FA(
        config,
        nn_model,
        fe)

    training_data_collector = FADataCollector()
    validation_data_collector = FADataCollector()

    # es = ESLookup(config,
    #               explorate=1300,
    #               fa=agent_fa)

    pa_nn_model = NNModel(
        PolicyGrader(),
        'adam',
        0.0001, # alpha
        0.1, # regularization constant
        512, # batch_size
        MAX_PA_ITERATIONS)

    sa_pa = SA_PA(config, pa_nn_model, fe)
    es = ESPolicy(config,
                  fa=agent_fa,
                  pa=sa_pa)

    explorer = Explorer(config, es)
    
    fa_learner = th.create_agent(config, 
                    alg = 'qlambda',
                    lam = 0.8,
                    fa=agent_fa)
    pa_learner = PolicyLearner(sa_pa, agent_fa)
    
    # ------------------ Training -------------------

    #test_agent = FAAgent(config, agent_fa)
    test_agent = FAExplorer(config, ESBest(config, agent_fa)) #TODO: should be pa-based best
    evaluator = Evaluator(config, episode_factory, test_agent, agent_fa)

    if False: # to train/test without exploration and processing
        util.pre_agent_alg = agent_fa.prefix()
# 
#         if False: # to train new model on dataset
#             dir = "791563_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i30000_FFEelv__NNconvnet_lookab5__"
#             agent_fa.init_net(default_net)
#             training_data_collector.load_dataset(dir, "final_t")
#             validation_data_collector.load_dataset(dir, "final_v")
#             agent_fa.train(training_data_collector, validation_data_collector)
#             agent_fa.report_stats()
#         elif False: # to train existing model on dataset
#             dir = "543562_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.0005_r0.5_b512_i1500_FFEelv__NNconvnet_lookab5__"
#             agent_fa.load_model(dir, "v3")
#             training_data_collector.load_dataset(dir, "final_t")
#             validation_data_collector.load_dataset(dir, "final_v")
#             agent_fa.train(training_data_collector, validation_data_collector)
#             agent_fa.report_stats()
#         elif True: #If load model params
#             agent_fa.init_net(default_net)
#             dir = "569160_Coindrop_DR_neural_bound_a0.002_r0.01_b512_i400_FFEelv__NNconvnet__"
#             agent_fa.load_model_params(dir, "v3")
#         elif False: #If load model architecture (and classes)
#             dir = "569160_Coindrop_DR_neural_bound_a0.002_r0.01_b512_i400_FFEelv__NNconvnet__"
#             agent_fa.load_model(dir, "v3")
            
        #agent_fa.save_model("v3")
        #evaluator.run_test(10)
    
    elif True: # If Run episodes
        
        trainer = EpochTrainer(episode_factory, [explorer], fa_learner, 
                               pa_learner,
                               training_data_collector,
                               validation_data_collector,
                               evaluator,
                               explorer.prefix() + "_" + fa_learner.prefix())
        
        if True:
            # To start training afresh 
            agent_fa.init_default_model()
            sa_pa.init_default_model()
            
#         elif False:
#             # To start fresh but using existing episode history / exploration
#             dir = "791563_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i30000_FFEelv__NNconvnet_lookab5__"
#             nn_model.init_net(default_net)
#             explorer.load_episode_history("explorer", dir)
#             es.load_exploration_state(dir)
#             opponent.load_episode_history("opponent", dir)
        elif False:
            # To start training from where we last left off.
            # i.e., load episodes history, exploration state, and FA model
            dir = "517637_RC2 rect_DR_elookup_q_lambda_g1.0_l0.80neural_a1e-06_r0.0001_b512_i50000_F1ffff_Cmse_Oadam__"
            explorer.load_episode_history("explorer", dir)
            es.load_exploration_state(dir)
            agent_fa.load_model(dir, "v3")
            trainer.load_stats(dir)
#         elif False:
#             # For single-epoch training/testing.
#             # Load last training dataset and model, but not earlier history
#             dir = "330041_Coindrop_DR_q_lambda_epat_l0.95neural_a0.0005_r0_b512_i1000_F_NNconvnetlook3__"
#             training_data_collector.load_dataset(dir, "final_t")
#             validation_data_collector.load_dataset(dir, "final_v")
#             agent_fa.load_model(dir, "v3")
    
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

