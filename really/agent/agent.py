'''
Created on 24 Apr 2019

@author: enerve
'''

class Agent(object):
    '''
    Base class for an agent that takes actions
    '''

    def init_episode(self, initial_state):#, initial_heights):
        ''' Initialize for a new episode
        '''
        self.G = 0
        self.moves = 0
        
    def see_outcome(self, reward, new_state, moves=0):
        ''' Observe the effects on this agent of an action taken - possibly by
            another agent.
        '''
        self.G += reward
        self.moves = moves
         
    def _choose_action(self):
        pass
         
    def next_action(self):
        ''' Agent's turn. Chooses the next move '''
        return self._choose_action()

    def episode_over(self):
        ''' Wrap up episode '''
        pass

    def learn_from_history(self):
        pass
    
    def process(self):
        ''' Process whatever happened so far
        '''
        pass

    def collect_stats(self, ep, num_episodes):
        pass
    
    def save_stats(self, pref=""):
        pass

    def load_stats(self, subdir, pref=""):
        pass

    def report_stats(self, pref):
        pass
    
    def live_stats(self):
        pass
