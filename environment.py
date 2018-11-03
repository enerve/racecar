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
    
    #     R_time_step = -1
    #     R_crash = 0
    #     R_progress = 5000
    #     R_finishline = 0
    #     R_milestone = 10
    R_time_step = -10
    R_crash = -200
    R_progress = 0
    R_finishline = 0
    R_milestone = 100
    
    def __init__(self, track, car, num_junctures, should_record=False):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.track = track
        self.car = car

        # Junctures are where actions can be taken
        self.num_junctures = num_junctures
        # Milestones are where rewards may be given
        self.num_milestones = track.num_milestones
        
        self.curr_juncture = 0
        self.curr_milestone = 0
        self.curr_time = 0

        self.car.restart(track, should_record)

        self.track_anchor = self.track.anchor(self.car.location)
        self.reached_finish = False
        self.last_progress = 0
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
            #self.logger.debug("Location: %s", (self.car.location, ))
            R += self.R_time_step
            self.track_anchor = self.track.anchor(self.car.location, self.track_anchor)
            if not self.track.is_inside(self.track_anchor):
                # Car has crashed!
                #self.logger.debug("Car crashed")
                # Think of this R as a penalty for untravelled track (plus 5000)
                R += self.R_crash
                R += self.R_progress * self.last_progress
                return (R, None)
            self.last_progress = self.track.progress_made(self.track_anchor)
            if not self.track.within_milestone(self.track_anchor, 
                                               self.curr_milestone,
                                               next_milestone):
                # milestone reached
                self.curr_milestone += 1
                #self.logger.debug("milestone %d reached", self.curr_milestone)
                next_milestone = (self.curr_milestone + 1) % self.num_milestones
                R += self.R_milestone
                #TODO: maybe there are multiple milestones to "jump"
            if not self.track.within_juncture(self.track_anchor, 
                                              self.curr_juncture,
                                              next_juncture):
                # Car location has moved beyond juncture
                # TODO: what if it has actually moved backwards?
                #self.logger.debug("juncture %d reached", next_juncture)
                break
        

        # Advance curr juncture to cover curr location
        while True:
            self.curr_juncture += 1
            next_juncture = (self.curr_juncture + 1) % self.num_junctures
            
            # Reached finish line?
            if self.curr_juncture == self.num_junctures:
                self.reached_finish = True
                #self.logger.debug("Reached finish line")
                R += self.R_finishline
                R += self.R_progress * 1 # 100%
                return (R, None)

            # if curr juncture now covers location, stop advancing juncture
            if self.track.within_juncture(self.track_anchor,
                                          self.curr_juncture,
                                          next_juncture):
                break
                
            # Skip through bypassed junctures
            #self.logger.debug("jumping juncture section %d", self.curr_juncture)
            
        return (R, self.state_encoding())
    
    def get_location(self):
        return self.car.location
    
    def has_reached_finish(self):
        return self.reached_finish
    
    def total_time_taken(self):
        return self.curr_time
            
    def state_encoding(self):
        if self.car is None:
            return None
        j = self.curr_juncture
        l = self.track.lane_encoding(self.track_anchor)
        v, d = self.car.state_encoding()
        return j, l, v, d

    def get_action_history(self):
        return self.car.action_history
    
    def report_history(self):
        if not self.should_record:
            return

        for vect, act in zip(self.car.vector_history, self.car.action_history):
            steer, accel = act
            dirn, sp = vect
            self.logger.info("  Steered %s with Accel %s, dirn %d, speed %d", 
                              ['L', '_', 'R'][steer], 
                              ['-1', '0', '+1'][accel],
                              dirn, sp)
        self.logger.info("  SteersA: %s", [a[0] for a in self.car.action_history])
        self.logger.info("  AccelsA: %s", [a[1] for a in self.car.action_history])
        self.logger.info("  Total time taken: %d", self.curr_time)

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
            self.car.draw(self.track.location_to_coordinates(loc), A)

            im = ax.imshow(A, aspect='equal')
#             ax.text(50, 50, "Testtttt", fontdict=None)
            debuginfo = ""#"S:%d\nD:%d" % (speed, dir)
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
    
    def show_path(self, save=True, show=True, env2=None, pref=""):
        if not self.should_record:
            return

        ax = plt.gca()
        plt.axis('off')
        A = self.track.draw()
        for pos in self.car.location_history:
            loc, speed, dir = pos
            self.car.draw(self.track.location_to_coordinates(loc), A)

        if env2:
            for pos in env2.car.location_history:
                loc, speed, dir = pos
                env2.car.draw(env2.track.location_to_coordinates(loc), A,
                              [0, 0, 200])

        ax.imshow(A, aspect='equal')

        if save:
            util.save_plot(pref)
        if show:
            plt.show()
        plt.close()