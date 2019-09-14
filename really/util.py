'''
Created on Sep 10, 2018

@author: enerve
'''

import logging
import numpy as np
import os
import time, datetime
import torch.onnx

import matplotlib.pyplot as plt
import matplotlib.animation as animation


use_gpu = False

pre_outputdir = None
pre_problem = None
pre_alg = None
pre_agent_alg = None
pre_student_alg = None
pre_tym = None
arg_bin_dir = None

def init(args):
    global pre_tym, pre_outputdir, arg_bin_dir
    pre_tym = str(int(round(time.time()) % 1000000))
    pre_outputdir = args.output_dir
    arg_bin_dir = args.bin

    init_gpu(args)

def init_gpu(args):
    global use_gpu
    if args.gpu and torch.cuda.is_available:
        use_gpu = True

def init_logger():
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.INFO)

    logger.debug("Starting at %s", datetime.datetime.now())
    if use_gpu:
        logger.debug("Cuda is available, using GPU as requested")

def prefix(other_tym=None):
    alg = ''
    if pre_agent_alg:
        alg += 'DR_%s_' % pre_agent_alg
    if pre_student_alg:
        alg += 'ST_%s_' % pre_student_alg
    
    return (pre_outputdir or '') + \
        ("%s_"%(other_tym or pre_tym)) + \
        ("%s_"%(pre_problem or '')) + \
        ("%s_"%alg)
        
def subdir(other_tym=None):
    return prefix(other_tym) + "/"


# ------ Drawing ---------

filenames_used = {}

def save_plot(pref=''):
    global filenames_used
    fname = prefix() + pref
    
    if fname in filenames_used:
        filenames_used[fname] += 1
        fname += "_%d" % filenames_used[fname]
    else:
        filenames_used[fname] = 1
    
    fname = fname + '.png'
    plt.savefig(fname, bbox_inches='tight')

def heatmap(P, extent, title=None, cmap='hot', aspect='auto', pref=''):
    ''' Plots a heatmap
        extent: (left, right, bottom, top)
    '''
    plt.imshow(P, cmap=cmap, interpolation='none', extent=extent, aspect=aspect)
    #plt.axis('off')
    if title is not None:
        plt.title(title)
    save_plot(pref)
    plt.show()

def start_interactive():
    plt.ion()

def stop_interactive():
    plt.ioff()
    plt.cla()

def plot(lines, x, labels=None, title=None, pref=None, ylim=None, live=False):
    if live:
        plt.cla()
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
    if live:
        plt.draw()
#         fig.canvas.start_event_loop(0.001)
        plt.pause(0.01)
    else:
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

def hist(x, bins=10, range=None, title=None, pref=None):
    plt.hist(x, bins, range)
    if title:
        plt.title(title, loc='center')
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
        
def save_hist_animation(dists, bins, range, ymax=None, title="", pref=""):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    plt.rcParams['animation.ffmpeg_path'] = arg_bin_dir + 'ffmpeg'
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='enerve'), bitrate=1800)
    def animate(x):
        ax1.clear()
        if ymax:
            ax1.set_ylim([0, ymax])
        ax1.set_title(title)
        ax1.hist(x, 100, range)
    ani = animation.FuncAnimation(fig, animate, dists, interval=10)
    ani.save(prefix() + pref + '.mp4', writer=writer)

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

def append(A, fname, subdir=None, suffix=None):
    fname = pre_outputdir \
        + ("%s/"%subdir if subdir else '') \
        + fname \
        + ("_%s"%suffix if suffix else '') \
        + '.npy'
    if os.path.isfile(fname):
        A_old = np.load(fname)
        logger.debug("%s", A_old)
        A = np.concatenate((A_old, A), axis=0)
    np.save(fname, A)

def torch_save(M, fname, suffix=None):
    os.makedirs(subdir(), exist_ok=True)
    fname = subdir() \
        + fname \
        + ("_%s"%suffix if suffix else '') #\
        #+ '.npy'
    torch.save(M, fname)

def torch_load(fname, subdir=None, suffix=None):
    fname = pre_outputdir \
        + ("%s/"%subdir if subdir else '') \
        + fname \
        + ("_%s"%suffix if suffix else '') #\
        #+ '.npy'
    return torch.load(fname)

def torch_export(M, dummy_input, fname, suffix=None):
    os.makedirs(subdir(), exist_ok=True)
    fname = subdir() \
        + fname \
        + ("_%s"%suffix if suffix else '') \
        + '.onnx'
    torch.onnx.export(M, dummy_input, fname)
