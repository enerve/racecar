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

    def __init__(self, track, car, num_milestones, should_record=False):
        '''
        Constructor
        '''
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.num_milestones = num_milestones
        
        self.track = track
        self.car = car
        self.car.restart(track, should_record)
        self.curr_milestone = 0
        self.curr_time = 0
        self.reached_finish = False
        #self.logger.debug("%s", self.state_encoding())
        
        self.should_record = should_record
        
    def step(self, A):
        ''' Performs a full step, using given actions
            Returns reward and next state
        '''
        
        #steer, accel = A
        #self.logger.debug("  Step: %d, %d", steer, accel)
        self.car.take_action(A)
        self.car.move()
        self.curr_time += 1
        R = -1


        next_milestone = (self.curr_milestone + 1) % self.num_milestones
        while self.track.is_inside(self.car.location) and \
                self.track.within_section(self.car.location, self.curr_milestone,
                                          next_milestone):
            #self.logger.debug("  moving further inside")
            self.car.move()
            self.curr_time += 1
            R += -1
        
        if not self.track.is_inside(self.car.location):
            #R += -100
            R += 5000 * self.track.progress_made(self.car.location)
    
            return (R, None)
        else:
            # milestone reached
            R += 10
            self.curr_milestone += 1
            if self.curr_milestone == self.num_milestones:
                self.reached_finish = True
                R += 5000 * 1 # 100 percent
                return (R, None)

            next_milestone = (self.curr_milestone + 1) % self.num_milestones
            while not self.track.within_section(self.car.location,
                                             self.curr_milestone,
                                             next_milestone):
                self.logger.debug("jumped section")
                self.curr_milestone += 1
                next_milestone = (self.curr_milestone + 1) % self.num_milestones
                #R += 100
                
        return (R, self.state_encoding())
    
    def has_reached_finish(self):
        return self.reached_finish
    
    def total_time_taken(self):
        return self.curr_time
            
    def state_encoding(self):
        if self.car is None:
            return None
        m = self.curr_milestone
        l = self.track.lane_encoding(self.car.location)
        v, d = self.car.state_encoding()
        return m, l, v, d

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