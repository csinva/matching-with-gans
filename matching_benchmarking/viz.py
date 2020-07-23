from os.path import join as oj

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from sklearn.utils.multiclass import unique_labels
from style import cb, cr, cg, cp
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
        '''
        means0 = np.array([np.mean(g0[k]) for k in ks])
        means1 = np.array([np.mean(g1[k]) for k in ks])
        args = np.argsort(np.abs(means0 - means1))
        '''
        args = np.arange(len(ks))
    
    for i, (g, lab) in enumerate(zip([g0, g1], ['Female', 'Male'])):
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
            plt.yticks(ys, [k[0].upper() + k[1:].replace('black', 'Black').replace('_', ' ')
                            for k in ks[args]])
        else:
            plt.yticks(ys, ['' for k in ks[args]])
#     plt.xlabel('Mean value in dataset')
    plt.grid()
    return args


def plot_confusion_matrix(y_true, y_pred, class_label,
                          ax,
                          normalize=False,
                         
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Params
    ------
    classes: np.ndarray(Str)
        classes=np.array(['aux-', 'aux+'])
    """
    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)

    
    ax.set(xlabel=class_label, ylabel=class_label)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax


def plot_subgroup_mean_diffs(ds, ks, k_group, figsize=None):
    if figsize is None:
        figsize=(12, 3)
    R, C = 1, 2
    fig = plt.figure(dpi=200, figsize=figsize)
    ks_g = [k for k in ks if not k == k_group]
    args = None
    colors = [cr, cb]
    titles = ['Before matching', 'After matching']

    for i, d in enumerate(ds):
        d = d[ks]

        # normalize to [0, 1]
    #     d = (d - d.min()) / (d.max() - d.min())
        g0 = d[d[k_group] == 0]
        g1 = d[d[k_group] == 1]

        ax = plt.subplot(R, C, i + 1)
        args = plot_subgroup_means(g0, g1,
                                    ks=np.array(ks_g),
                                    CI='wilson',
                                    ticklabels=i == 0, args=None, colors=colors)
        plt.title(titles[i])
        plt.xlim((0, 1))
    fig.text(0.5, 0, 'Mean fraction of points which have this attribute', ha='center')
    plt.legend(title='Perceived gender', bbox_to_anchor=(1, 0.5))  
