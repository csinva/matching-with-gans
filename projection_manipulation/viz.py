import os
from os.path import join as oj

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import pickle as pkl
sys.path.append('..')

import data
import style
import config
from config import *
import util
import glob
import json, jsonlines

df = data.load_all_labs()
df = df.set_index('fname_id')

# get fnames
fname_nps = [f for f in sorted(os.listdir(DIR_GEN)) if 'npy' in f] # these start at 00001
fname_ids = np.array([f[:-4] for f in fname_nps])
idxs_calculated = np.array([int(x) - 1 for x in fname_ids]) # this starts at 0

# trim df to only have the relevant ids
df = df.loc[fname_ids]


def hist_subgroups(means, labs, labs_list, BINS=4):
    for lab in labs_list:
        plt.hist(means[labs[lab]], BINS, alpha=0.5, label=lab)
    plt.ylabel('Count')
    plt.xlabel('Fraction of images said to be "Same"')
    plt.legend()
    plt.show()

def boxplot_subgroups(vals, labs, labs_list, confs='sem', ret=False, width=7):
    '''
    vals: array_like
        what to plot
    labs: dataframe / dict
        contains indexes over which to extract the vals
    labs_list: array_like
        labels for the means
    confs: array_like
        confidence intervals
    '''
    plt.figure(dpi=300, figsize=(width, 3))
    lists = [vals[labs[lab]] for lab in labs_list]
    ys = np.arange(1, len(labs_list) + 1)
    if confs is None:
        plt.boxplot(lists, vert=False, showmeans=True)
    else:
        means = [np.mean(l) for l in lists]
        if confs == 'sem':
            sems = [1.96 * np.std(l) / np.sqrt(len(l)) for l in lists]
        elif confs == 'wilson':
            wilsons = [util.wilson(l) for l in lists]
            sems = np.abs(np.array([[w[0], w[1]] for w in wilsons]).transpose())
        plt.errorbar(means,
                     ys, xerr=sems,
                     linestyle='None', marker='o', ms=8)
        # plt.boxplot(lists, vert=False, showmeans=True)
    plt.yticks(ys, [x.capitalize() for x in labs_list])
    plt.xlabel('Fraction of pairs labelled as "Same"')
    plt.ylabel('True conditions')
    plt.xlim((0.3, 1.05))
    if ret:
        return np.array(means), np.array(sems)
    
def annotators_num_plot(annotations):
    n_annotations = sorted(annotations.annotations, reverse=True)
    plt.figure(dpi=100)
    plt.grid()
    plt.plot(range(1, 1 + annotations.N_ANNOTATORS), n_annotations)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Number of annotators')
    plt.ylabel('Number of annotations')
    plt.title('Work of individual annotators')
    plt.show()    