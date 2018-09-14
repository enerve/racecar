'''
Created on Sep 6, 2018

@author: enerve
'''

import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import util


class Environment(object):
    '''
    Environment is instantiated for a 1-episode run of the given Car on the
    given Track. It is responsible for interpreting a driver's action,
    for calculating rewards based on resulting Car states and predefined
    milestones.
    It can also convert the Car's reality back a simplified "state" encoding
    for the Driver.
    '''
    
    NUM_MILESTONES = 27   # Points at which rewards will be given

    def __init__(self, track, car, num_junctures, should_record=False):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.num_junctures = num_junctures
        self.num_milestones = Environment.NUM_MILESTONES
        
        self.track = track
        self.car = car
        self.car.restart(track, should_record)
        self.curr_juncture = 0
        self.curr_milestone = 0
        self.curr_time = 0
        self.reached_finish = False
        #self.logger.debug("%s", self.state_encoding())
        
        self.should_record = should_record
        
    def step(self, A):
        ''' Performs a full step, using given actions
            Returns reward and next state
        '''
        
        self.car.take_action(A)

        next_juncture = (self.curr_juncture + 1) % self.num_junctures                
        next_milestone = (self.curr_milestone + 1) % self.num_milestones

        # Run physics until next juncture, collecting rewards along the way
        R = 0
        while True:
            #self.logger.debug("  moving further inside")
            self.car.move()
            self.curr_time += 1
            R += -1
            if not self.track.within_milestone(self.car.location, 
                                               self.curr_milestone,
                                               next_milestone):
                # milestone reached
                self.curr_milestone += 1
                next_milestone = (self.curr_milestone + 1) % self.num_milestones
                R += 10
            if not self.track.is_inside(self.car.location):
                # Car has crashed!
                # Think of this R as a penalty for untravelled track (plus 5000)
                R += 5000 * self.track.progress_made(self.car.location)
        
                return (R, None)
            if not self.track.within_juncture(self.car.location, 
                                              self.curr_juncture,
                                              next_juncture):
                # Car location has moved beyond juncture
                break
        

        # Advance curr juncture to cover curr location
        while True:
            self.curr_juncture += 1
            next_juncture = (self.curr_juncture + 1) % self.num_junctures
            
            # Reached finish line?
            if self.curr_juncture == self.num_junctures:
                self.reached_finish = True
                R += 5000 * 1 # 100 percent
                return (R, None)

            # if curr juncture now covers location, stop advancing juncture
            if self.track.within_juncture(self.car.location,
                                          self.curr_juncture,
                                          next_juncture):
                break
                
            # Skip through bypassed junctures
            self.logger.debug("jumping section")
            
        return (R, self.state_encoding())
    
    def has_reached_finish(self):
        return self.reached_finish
    
    def total_time_taken(self):
        return self.curr_time
            
    def state_encoding(self):
        if self.car is None:
            return None
        j = self.curr_juncture
        l = self.track.lane_encoding(self.car.location)
        v, d = self.car.state_encoding()
        return j, l, v, d

    def report_history(self):
        if not self.should_record:
            return

        for i, act in enumerate(self.car.action_history):
            steer, accel = act
            dirn, sp = self.car.vector_history[i]
            self.logger.debug("  Steered %d with Accel %d, dirn %d, speed %d", 
                              steer-1, accel-1, dirn, sp)
        self.logger.debug("  SteersA: %s", [a[0] for a in self.car.action_history])
        self.logger.debug("  AccelsA: %s", [a[1] for a in self.car.action_history])
        self.logger.debug("  Total time taken: %d", self.curr_time)

    def play_movie(self, save=True, show=True, pref=""):
        if not self.should_record:
            return

        fig = plt.figure()
        ax = plt.gca()
        plt.axis('off')
        image_list = []
        for pos in self.car.location_history:
            loc, speed, dir = pos
            A = self.track.draw()
            self.car.draw(loc, A)

            im = ax.imshow(A, cmap='Greys')#, animated=True)
#             ax.text(50, 50, "Testtttt", fontdict=None)
            debuginfo = "S:%d\nD:%d" % (speed, dir)
            t = ax.annotate(debuginfo, (10,20)) # add text
            
            image_list.append([im, t])
        #plt.show()
        ani = animation.ArtistAnimation(fig, image_list,
                                        interval=50, blit=True,
                                        repeat_delay=1000)
        if show:
            plt.show()
        if save:
            plt.rcParams['animation.ffmpeg_path'] = u'/Users/erw/miniconda2/envs/my_env/bin/ffmpeg'
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='enerve'), bitrate=1800)
    
            logging.getLogger("matplotlib.animation").setLevel(logging.INFO)
            ani.save(util.prefix() + pref + '.mp4', writer=writer)
        plt.close()