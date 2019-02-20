'''
Created on Sep 10, 2018

@author: enerve
'''

import logging
import numpy as np
import os
import time
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation

logger = logging.getLogger(__name__)

def init_logger():
    #logger.setLevel(logging.INFO)
    pass

use_gpu = False

def init_gpu(args):
    global use_gpu
    if args.gpu and torch.cuda.is_available:
        print("Cuda is available, using GPU as requested")
        use_gpu = True

# ------ Drawing ---------

pre_outputdir = None
pre_problem = None
pre_alg = None
pre_driver_alg = None
pre_student_alg = None
pre_tym = None
arg_bin_dir = None

def init(args):
    global pre_tym, pre_outputdir, arg_bin_dir
    pre_tym = str(int(round(time.time()) % 1000000))
    pre_outputdir = args.output_dir
    arg_bin_dir = args.bin

    init_gpu(args)

    logging.getLogger("matplotlib").setLevel(logging.INFO)

def prefix(other_tym=None):
    alg = ''
    if pre_driver_alg:
        alg += 'DR_%s_' % pre_driver_alg
    if pre_student_alg:
        alg += 'ST_%s_' % pre_student_alg
    
    return (pre_outputdir or '') + \
        ("%s_"%(other_tym or pre_tym)) + \
        ("%s_"%(pre_problem or '')) + \
        ("%s_"%alg)
        
def subdir(other_tym=None):
    return prefix(other_tym) + "/"

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

def plot(lines, x, labels=None, title=None, pref=None, ylim=None):
    for i in range(len(lines)):
        l = lines[i]
        if labels:
            plt.plot(x, l, label=labels[i])
        else:
            plt.plot(x, l)
    if labels:
        plt.legend()
    if title:
        plt.title(title, loc='center')
    if ylim:
        plt.ylim(ylim)
    save_plot(pref)
    plt.show()

def plot_all(lines, xs, labels=None, title=None, pref=None, ylim=None):
    for i in range(len(lines)):
        l = lines[i]
        x = xs[i]
        if labels:
            plt.plot(x, l, label=labels[i])
        else:
            plt.plot(x, l)
    if labels:
        plt.legend()
    if title:
        plt.title(title, loc='center')
    if ylim:
        plt.ylim(ylim)
    save_plot(pref)
    plt.show()
    
def draw_image(A):
    plt.imshow(A)
    plt.show()
    
def scatter(x, y, values, xlabel, ylabel, pref=None):
    plt.scatter(x, y, c=values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_plot(pref)
    plt.show()

class Plotter:
    def __init__(self, title):
        self.title = title
        self.I_list = []

    def add_image(self, P, debuginfo=""):
        self.I_list.append((P, debuginfo))
        
    def save(self, fname, pref=None):
        dump(np.asarray([im[0] for im in self.I_list], dtype=np.float),
             fname + "_I", pref)
        dump(np.asarray([im[1] for im in self.I_list], dtype=np.str),
             fname + "_D", pref)

    def load(self, fname, subdir, pref=None):
        ims = load(fname + "_I", subdir, pref)
        debugs = load(fname + "_D", subdir, pref)
        self.I_list = list(zip(ims, debugs))

    def play_animation(self, cmap='hot', vmin=None, vmax=None, 
                        show=True, save=True, pref=""):
        fig = plt.figure()
        image_list = []
        ax = plt.gca()
        for I in self.I_list:
            P, debug = I
            im = ax.imshow(P, cmap=cmap, interpolation='nearest',
                           vmin=vmin, vmax=vmax)
            _ = plt.title(self.title)
            t = ax.annotate(debug, (10,20)) # add text
            image_list.append([im, t])

        ani = animation.ArtistAnimation(fig, image_list,
                                        interval=50, blit=True,
                                        repeat_delay=1000)
        if show:
            plt.show()

        if save:
            plt.rcParams['animation.ffmpeg_path'] = arg_bin_dir + 'ffmpeg'
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='enerve'), bitrate=1800)
            
            logging.getLogger("matplotlib.animation").setLevel(logging.INFO)
            logging.getLogger("matplotlib.axes").setLevel(logging.INFO)
            ani.save(prefix() + pref + '.mp4', writer=writer)
        plt.close()

def checkpoint_reached(ep, checkpoint_divisor):
    return checkpoint_divisor > 0 and \
        ep > 0 and ep % checkpoint_divisor == 0

# ------ Storing run data ---------

def dump(A, fname, suffix=None):
    os.makedirs(subdir(), exist_ok=True)
    fname = subdir() \
        + fname \
        + ("_%s"%suffix if suffix else '') \
        + '.npy'
    np.save(fname, A)

def load(fname, subdir=None, suffix=None):
    fname = pre_outputdir \
        + ("%s/"%subdir if subdir else '') \
        + fname \
        + ("_%s"%suffix if suffix else '') \
        + '.npy'
    return np.load(fname)
