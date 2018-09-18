'''
Created on Sep 10, 2018

@author: enerve
'''

import logging
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.animation as animation

logger = logging.getLogger(__name__)

def init_logger():
    #logger.setLevel(logging.INFO)
    pass


# ------ Drawing ---------

pre_outputdir = None
pre_problem = None
pre_alg = None
pre_tym = None

def prefix_init(args):
    global pre_tym, pre_outputdir
    pre_tym = str(int(round(time.time()) % 1000000))
    pre_outputdir = args.output_dir

def prefix(other_tym=None):
    return (pre_outputdir if pre_outputdir else '') + \
        ("%s_"%pre_problem if pre_problem else '') + \
        ("%s_"%pre_alg if pre_alg else '') + \
        ("%s_"%(other_tym or pre_tym))

def save_plot(pref=None):
    fname = prefix() \
        + ("%s_"%pref if pref else '') \
        + '.png'
    plt.savefig(fname, bbox_inches='tight')

def heatmap(P, extent, title=None, cmap='hot', pref=None):
    plt.imshow(P, cmap=cmap, interpolation='none', extent=extent, aspect='auto')
    #plt.axis('off')
    if title is not None:
        plt.title(title)
    save_plot(pref)
    plt.show()

def plot(lines, x, labels, title=None, pref=None):
    for i in range(len(lines)):
        l = lines[i]
        plt.plot(x, l, label=labels[i])
    plt.legend()
    if title:
        plt.title(title, loc='center')
    save_plot(pref)
    plt.show()

def plot_all(lines, xs, labels, title=None, pref=None):
    for i in range(len(lines)):
        l = lines[i]
        x = xs[i]
        plt.plot(x, l, label=labels[i])
    plt.legend()
    save_plot(pref)
    plt.show()

class Plotter:
    def __init__(self, title):
        self.title = title
        self.I_list = []

    def add_image(self, P, debuginfo=""):
        self.I_list.append((P, debuginfo))
        
    def play_animation(self, cmap='hot', vmin=None, vmax=None, 
                        show=True, save=True, pref=""):
        fig = plt.figure()
        image_list = []
        ax = plt.gca()
        for I in self.I_list:
            P, debug = I
            im = ax.imshow(P, cmap=cmap, interpolation='nearest',
                           vmin=vmin, vmax=vmax)
            plt.title(self.title)
            t = ax.annotate(debug, (10,20)) # add text
            image_list.append([im, t])

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
            ani.save(prefix() + pref + '.mp4', writer=writer)
        plt.close()

# ------ CSVs ---------

def dump(A, pref):
    fname = prefix() \
        + ("%s_"%pref if pref else '') \
        + '.csv'
    np.savetxt(fname, A, delimiter=",")

def load(fname):
    fname = (pre_outputdir if pre_outputdir else '') + \
            fname
    return np.genfromtxt(fname, delimiter=',')
