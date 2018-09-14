'''
Created on Sep 5, 2018

@author: enerve
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':

    fig = plt.figure()
    
    
#     def f(x, y):
#         return np.sin(x) + np.cos(y)
    
    def circle():
        H = 100
        W = 120 
        R = 40
        x = np.linspace(-W/2, W/2, W)
        y = np.linspace(-H/2, H/2, H).reshape(-1, 1)
        Circ = (x * x + y * y)
        T = 100 * np.logical_and(Circ < R*R, Circ > R*R/3)
    
        return T

    def f(Track, a, b):
        T = Track
        T[a, b] = 200
        col = 0
        return T
    
    Track = circle()
    a = 30
    b = 25

    NUM_VELOCITIES = 10
    NUM_DIRECTIONS = 40
    
    print("%s" % (Track.shape[0] * Track.shape[1] * NUM_VELOCITIES * NUM_DIRECTIONS))
    
    #S = np.zeros(Track.shape[0], Track.shape[1], NUM_VELOCITIES, NUM_DIRECTIONS)
    
    
    
    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(20):
        a += 1
        b += 1
        im = plt.imshow(f(Track, a, b), cmap='Greys', animated=True)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    
    
    
    # ani.save('dynamic_images.mp4')
    
    plt.show()