'''
Created on Sep 1, 2019

@author: enerve
'''

from really.episode_factory import EpisodeFactory as EF
from .episode import Episode

class EpisodeFactory(EF):

    def __init__(self, config, track, car):
        self.config = config
        self.track = track
        self.car = car

    def new_episode(self, explorer_list):
        driver = explorer_list[0]
        return Episode(self.config, driver, self.track, self.car)