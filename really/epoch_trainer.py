'''
Created on Nov 6, 2018

@author: enerve
'''

import logging
import time

from really import util

class EpochTrainer:
    ''' A class that helps train the RL agent in stages, collecting episode
        history for an epoch and then training on that data.
    '''

    def __init__(self, episode_factory, explorer_list, learner, training_data_collector,
                 validation_data_collector, evaluator, prefix):
        self.episode_factory = episode_factory
        self.explorer_list = explorer_list #TODO: dict with keys
        self.explorer = explorer_list[0]
        self.learner = learner
        self.training_data_collector = training_data_collector
        self.validation_data_collector = validation_data_collector
        self.evaluator = evaluator
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
    
        util.pre_agent_alg = prefix
        self.logger.debug("Agent: %s", util.pre_agent_alg)

        self.stat_e_1000 = []
        
        self.ep = 0

    def train(self, num_episodes_per_epoch, num_epochs, num_explorations = 1,
              debug_run_first_epoch_data_only = False):
        
        total_episodes = num_episodes_per_epoch * num_epochs * num_explorations
           
        self.logger.debug("Starting for %d episodes x %d epochs x %d expls",
                          num_episodes_per_epoch, num_epochs, num_explorations)
        start_time = time.clock()
        
        totaltime_explore = 0
        totaltime_process = 0
        totaltime_train = 0
        
        #self.end_states = {}

        # Evaluate starting performance
        self.evaluator.evaluate(0)

        ep = ep_s = self.ep
        for expl in range(num_explorations):
            for epoch in range(num_epochs):
                # In each epoch, we first collect experience, then (re)train FA
                self.logger.debug("====== Expl %d epoch %d =====", expl, epoch)
                
                start_explore_time = time.clock()
                
                if not debug_run_first_epoch_data_only or epoch == 0:
                    # Run games to collect new data               
                    for ep_ in range(num_episodes_per_epoch):
                        # Create racecar_episode for a single episode
                        episode = self.episode_factory.new_episode(self.explorer_list)
                        
                        episode.run()
    
                        if ep_ % 10 == 0:
                            # Save for validation
                            self.explorer.save_episode_for_testing()
                            #TODO: self.opponent.save_game_for_testing()
                        else:
                            # Use for training
                            self.explorer.save_episode_for_training()
                            #TODO: self.opponent.save_game_for_training()
    
                        #stS = np.array2string(S, separator='')
                        #if stS in self.end_states:
                        #    self.end_states[stS] += 1
                        #else:
                        #    self.end_states[stS] = 1
    
                        self.explorer.collect_stats(episode,
                                                    ep, total_episodes)
                        #TODO: self.opponent.collect_stats(episode, ep, total_episodes)
                
                        if util.checkpoint_reached(ep, 1000):
                            self.stat_e_1000.append(ep)
                            self.logger.debug("Ep %d ", ep)

                        ep += 1

                    self.training_data_collector.reset_dataset()
                    self.validation_data_collector.reset_dataset()

                else:
                    # Reuse existing data
                    self.training_data_collector.replay_dataset()
                    self.validation_data_collector.replay_dataset()

                start_process_time = time.clock()
                totaltime_explore += (start_process_time - start_explore_time)
                
                self.logger.debug("-------- processing ----------")
                self.logger.debug("Learning from explorer history")
                self.learner.process(self.explorer.get_episodes_history(),
                                     self.training_data_collector,
                                     "explorer train")
                self.training_data_collector.report_collected_dataset()
                #self.learner.plot_last_hists()
                self.learner.process(self.explorer.get_test_episodes_history(),
                                     self.validation_data_collector,
                                     "explorer val")

#                 self.logger.debug("Learning from opponent history")
#                 self.learner.process(self.opponent.get_episodes_history(),
#                                      self.training_data_collector,
#                                      "opponent train")
#                 #self.learner.plot_last_hists()
#                 #self.learner.collect_last_hists()
#                 self.learner.process(self.opponent.get_test_episodes_history(),
#                                      self.validation_data_collector,
#                                      "opponent val")

                start_training_time = time.clock()
                totaltime_process += (start_training_time - start_process_time)

                self.logger.debug("-------- training ----------")
                self.learner.learn(self.training_data_collector,
                                   self.validation_data_collector)

                totaltime_train += (time.clock() - start_training_time)
                
                # Sacrifice some data for the sake of GPU memory
                if len(self.explorer.get_episodes_history()) >= 15000:#20000
                    #TODO: process opponent as well
                    self.logger.debug("Before: %d",
                                      len(self.explorer.get_episodes_history()))
                    self.explorer.decimate_history()
                    #self.opponent.decimate_history()
                    self.logger.debug("After: %d", 
                                      len(self.explorer.get_episodes_history()))
                
                self.evaluator.evaluate(ep)

                self.logger.debug("  Clock: %d seconds", time.clock() - start_time)

            #self.explorer.restart_exploration(1)

        self.logger.debug("Completed training in %0.1f minutes", (time.clock() - start_time)/60)
        self.logger.debug("   Total time for Explore: %0.1f minutes", (totaltime_explore)/60)
        self.logger.debug("   Total time for Process: %0.1f minutes", (totaltime_process)/60)
        self.logger.debug("   Total time for Train:   %0.1f minutes", (totaltime_train)/60)
    
        self.ep = ep
        

    def load_from_file(self, subdir):
        self.learner.load_model(subdir)
        #self.load_stats(subdir)

    def save_to_file(self, pref=''):
        # save learned values to file
        self.learner.save_model(pref=pref)
        
        # save stats to file
        self.save_stats(pref=pref)        
    
    def save_stats(self, pref=""):
        self.explorer.save_stats(pref="a_" + pref)
        #TODO: self.opponent.save_stats(pref="o_" + pref)
        self.learner.save_stats(pref="l_" + pref)
        self.evaluator.save_stats(pref="t_" + pref)

        self.learner.save_hists(["explorer train"])#TODO: , "opponent train"])
        self.learner.write_hist_animation("explorer train")
        
        
    def load_stats(self, subdir, pref=""):
        self.logger.debug("Loading stats...")

        self.explorer.load_stats(subdir, pref="a_" + pref)
        #TODO: self.opponent.load_stats(subdir, pref="o_" + pref)
        self.learner.load_stats(subdir, pref="l_" + pref)
        self.evaluator.load_stats(subdir, pref="t_" + pref)
    
        #self.learner.load_hists(subdir)
    
    def report_stats(self, pref=""):
        self.explorer.report_stats(pref="a_" + pref)
        #TODO: self.opponent.report_stats(pref="o_" + pref)
        self.learner.report_stats(pref="l_" + pref)
        self.evaluator.report_stats(pref="t_" + pref)
        
