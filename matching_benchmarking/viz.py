from os.path import join as oj

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import util

def plot_subgroup_means(g0, g1, ks, ticklabels=True, args=None,
                        colors=None, CI='sem'):
    '''Plots means (with errbar) horizontally for each subgroup
    args is used to ensure that yticks are put in same order
    g0: dataframe or dict
        each key will be one row on the y axis
        each val should be an np array whose mean and std will be plotted
    g1: dataframe or dict
        same as g0 for another group
    '''
    if 'pandas' in str(type(g0)):
        g0 = g0[ks]
        g1 = g1[ks]
    
    if args is None:
        means0 = np.array([np.mean(g0[k]) for k in ks])
        means1 = np.array([np.mean(g1[k]) for k in ks])
        args = np.argsort(np.abs(means0 - means1))
    
    for i, (g, lab) in enumerate(zip([g0, g1], ['Perceived as female', 'Perceived as male'])):
        lists = [g[k] for k in ks]
        means = np.array([np.mean(l) for l in lists])
        sems = np.array([np.std(l) / np.sqrt(l.size) for l in lists])
        if CI == 'sem':
            sems = 1.96 * sems
            sems = sems[args]
        elif CI == 'wilson':
            wilsons = [util.wilson(l) for l in lists]
            sems = np.abs(np.array([[w[0], w[1]] for w in wilsons]).transpose())
            sems = sems[:, args]
        ys = np.arange(len(ks))
        plt.errorbar(means[args], ys, label=lab, xerr=sems,
                     linestyle='', marker='.', markersize=10, color=colors[i])
        if ticklabels:
            plt.yticks(ys, [k.capitalize().replace('_', ' ') for k in ks[args]])
        else:
            plt.yticks(ys, ['' for k in ks[args]])
#     plt.xlabel('Mean value in dataset')
    plt.grid()
    return args


def plot_subgroup_diffs(g0, g1, ks, ticklabels=True, args=None,
                        colors=None, CI='sem'):
    '''Plots means (with errbar) horizontally for each subgroup
    args is used to ensure that yticks are put in same order
    g0: dataframe or dict
        each key will be one row on the y axis
        each val should be an np array whose mean and std will be plotted
    g1: dataframe or dict
        same as g0 for another group
    '''
    if 'pandas' in str(type(g0)):
        g0 = g0[ks]
        g1 = g1[ks]
    
    if args is None:
        means0 = np.array([np.mean(g0[k]) for k in ks])
        means1 = np.array([np.mean(g1[k]) for k in ks])
        args = np.argsort(np.abs(means0 - means1))
    
    for i, (g, lab) in enumerate(zip([g0, g1], ['Perceived as female', 'Perceived as male'])):
        lists = [g[k] for k in ks]
        means = np.array([np.mean(l) for l in lists])
        sems = np.array([np.std(l) / np.sqrt(l.size) for l in lists])
        if CI == 'sem':
            sems = 1.96 * sems
            sems = sems[args]
        elif CI == 'wilson':
            wilsons = [util.wilson(l) for l in lists]
            sems = np.abs(np.array([[w[0], w[1]] for w in wilsons]).transpose())
            sems = sems[:, args]
        ys = np.arange(len(ks))
        plt.errorbar(means[args], ys, label=lab, xerr=sems,
                     linestyle='', marker='.', markersize=10, color=colors[i])
        if ticklabels:
            plt.yticks(ys, [k.capitalize().replace('_', ' ') for k in ks[args]])
        else:
            plt.yticks(ys, ['' for k in ks[args]])
#     plt.xlabel('Mean value in dataset')
    plt.grid()
    return args