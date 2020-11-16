import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

import util
from style import cb, cr, cp, cpink


def plot_subgroup_means(g0, g1, ks, ticklabels=True, args=None, ms=10,
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
                     linestyle='', marker='.', markersize=ms, color=colors[i])
        if ticklabels:
            ylabs = [k[0].upper() + k[1:].replace('black', 'Black').replace('_', ' ')
                            for k in ks[args]]
            ylabs = [yl.replace('-Race=', 'Race≠').replace('Eyeglasses', 'Glasses') for yl in ylabs]
            plt.yticks(ys, ylabs)
        else:
            plt.yticks(ys, ['' for k in ks[args]])
    plt.grid(which='both')
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


def plot_subgroup_mean_diffs(ds, ks, k_group, figsize=None, vert=False, ms=10,
                             titles=['Before matching', 'After matching'], legend_loc='right'):
    '''Plots means of different subgroups horizontally
    Params
    ------
    ds: list of dataframes
        each element of ds is a dataframe to be plotted
    '''
    if figsize is None:
        figsize=[12, 3]
    R, C = 1, len(ds)
    if vert:
        R, C = C, R
    fig = plt.figure(dpi=200, figsize=figsize)
    ks_g = [k for k in ks if not k == k_group]
    args = None
    colors = [cpink, cb]

    lets = ['A', 'B', 'C']
    for i, d in enumerate(ds):
        d = d[ks]

        # normalize to [0, 1]
        #     d = (d - d.min()) / (d.max() - d.min())
        g0 = d[d[k_group] == 0]
        g1 = d[d[k_group] == 1]

        ax = plt.subplot(R, C, i + 1)
        plt.xticks([0, 0.5, 1])
        args = plot_subgroup_means(g0, g1,
                                   ks=np.array(ks_g),
                                   CI='wilson',
                                   ms=ms,
                                   ticklabels=i == 0, args=None, colors=colors)
        plt.title(titles[i])
        plt.title(lets[i], loc='left', fontweight='bold')
        plt.xlim((-0.1, 1.1))
        
    
    if legend_loc == 'right':
        plt.legend(title='Perceived gender', bbox_to_anchor=(1, 0.5), title_fontsize=16)
        fig.text(0.5, 0, 'Fraction of points with this attribute', ha='center', fontsize=16)
    elif legend_loc == 'bottom':
        for i in range(len(ds)):
            plt.subplot(R, C, i + 1)
            plt.ylim(-0.5, len(ks_g) - 0.5)
        plt.legend(title='Perceived gender', bbox_to_anchor=(-1.3, 0), loc='upper right', title_fontsize=16)
        fig.text(0.5, 0.05, 'Fraction of points\n with this attribute', ha='center', fontsize=16)
        