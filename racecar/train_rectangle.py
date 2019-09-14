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
from really.function import *
from really import cmd_line
from really import log
from really import util

from racecar.car import Car
from racecar.track import LineTrack
from racecar.episode_factory import EpisodeFactory
from racecar.rectangle_feature_eng import RectangleFeatureEng
from racecar.racecar_es_lookup import RacecarESLookup
from racecar.racecar_explorer import RacecarExplorer
from racecar.evaluator import Evaluator
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
    
    NUM_NEW_EPISODES = 4000
    NUM_EPOCHS = 10
    
    logger.debug("NUM_NEW_EPISODES=%d\t NUM_EPOCHS=%d", NUM_NEW_EPISODES, NUM_EPOCHS)
        
    episode_factory = EpisodeFactory(config, track, car)

    rect_fe = RectangleFeatureEng(config)
        
    h1 = 2000
    h2 = 2000
    h3 = 2000
    default_net = nn.Sequential(
        nn.Linear(rect_fe.num_inputs, h1),
        nn.Sigmoid(),
        nn.Linear(h1, h2),
        nn.Sigmoid(),
#         nn.Linear(h2, h3),
#         nn.Sigmoid(),
        nn.Linear(h3, rect_fe.num_actions()))
#     default_net = Net(4, 500, rect_fe.num_actions())

    agent_fa = NN_FA(
        'mse',
        'adam',
        0.000001, # alpha
        0.001, # regularization constant
        512, # batch_size
        4000, # max_iterations
        rect_fe)

    training_data_collector = FADataCollector(agent_fa)
    validation_data_collector = FADataCollector(agent_fa)

    es = RacecarESLookup(config,
                  explorate=1300,
                  fa=agent_fa)
    explorer = RacecarExplorer(config, es)
    
    learner = th.create_agent(config, 
                    alg = 'qlambda',
                    lam = 0.8,
                    fa=agent_fa)
    
    # ------------------ Training -------------------

    test_agent = FAAgent(config, agent_fa)
    evaluator = Evaluator(episode_factory, test_agent)

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
        
        trainer = EpochTrainer(episode_factory, [explorer], learner, 
                               training_data_collector,
                               validation_data_collector,
                               evaluator,
                               explorer.prefix() + "_" + learner.prefix())
        
        if True:
            # To start training afresh 
            agent_fa.init_net(default_net)
#         elif False:
#             # To start fresh but using existing episode history / exploration
#             dir = "791563_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i30000_FFEelv__NNconvnet_lookab5__"
#             agent_fa.init_net(default_net)
#             explorer.load_episode_history("agent", dir)
#             es.load_exploration_state(dir)
#             opponent.load_episode_history("opponent", dir)
#         elif False:
#             # To start training from where we last left off.
#             # i.e., load episodes history, exploration state, and FA model
#             #dir = "831321_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i25000_FFEelv__NNconvnet_lookab5__"
#             #dir = "789016_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.01_b512_i8000_FFEelv__NNconvnet_lookab5__"
#             dir = "178719_Coindrop_DR_eesp_sarsa_lambda_g0.9_l0.95neural_bound_a0.002_r0.001_b512_i150000_FFEelv__NNconvnet_lookab5__"
#             explorer.load_episode_history("agent", dir)
#             es.load_exploration_state(dir)
#             opponent.load_episode_history("opponent", dir)
#             agent_fa.load_model(dir, "v3")
#             #trainer.load_stats(dir)
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

    else: # load model, export to onnx
        
#         import onnx
#         model = onnx.load("/home/erwin/MLData/RL/output/655236_Coindrop_DR_q_lambda_epat_l0.95neural_a0.0005_r0_b512_i1000_FFEv2__NNconvnetlook3__/coindropV2.onnx")
#         onnx.checker.check_model(model)
#         print(onnx.helper.printable_graph(model.graph))
        
        dir = "986337_Coindrop_DR_sarsa_lambda_eesp_l0.95neural_bound_a0.0005_r0.5_b512_i500_FBAM__NNconvnetlookab5__"
        agent_fa.load_model(dir, "v3")
        
        agent_fa.export_to_onnx("v3")
        #agent_fa.viz()
        
    
if __name__ == '__main__':
    main()

