'''
Created on Sep 12, 2018

@author: enerve
'''

import logging

class Track:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
    def get_starting_position(self):
        self.logger.warning("Base method called")
    def is_inside(self, loc):
        self.logger.warning("Base method called")
    def progress_made(self, loc):
        self.logger.warning("Base method called")
    def within_juncture(self, loc, juncture_a, juncture_b):
        self.logger.warning("Base method called")
    def within_milestone(self, loc, milestone_a, milestone_b):
        self.logger.warning("Base method called")
    def lane_encoding(self, loc):
        self.logger.warning("Base method called")
    def coordinates_to_location(self, coordinates):
        self.logger.warning("Base method called")
    def location_to_coordinates(self, location):
        self.logger.warning("Base method called")
    def draw(self):
        self.logger.warning("Base method called")
