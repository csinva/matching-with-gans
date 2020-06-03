import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import datetime
import time
import jsonlines
from PIL import Image
from scipy.optimize import curve_fit
import scipy as sy
import pickle as pkl
import seaborn as sns
from os.path import join as oj
import sklearn.model_selection
import sklearn.linear_model
import numpy.linalg as npl
import pandas as pd
import viz
from tqdm import tqdm
from scipy import optimize
import scipy.stats
import torch
import models
import util
import style
import losses

PROCESSED_DIR = 'processed'
RESULTS_DIR = 'results'
DIRECTIONS_DIR = '../data/annotation-dataset-stylegan2/linear_models/new' # path to many datasets, includes on directory before the below dirs
GENERATING_LATENTS_DIR = '../data/annotation-dataset-stylegan2/data'

def plot_mse_vs_corrs():
    # linear plots
    df = pd.read_pickle(oj(PROCESSED_DIR, '06_df_loss_tradeoff_linear.pkl'))
    df = df.iloc[1:]
    df = df.iloc[:-4]
    R, C = 1, 2
    plt.figure(figsize=(12, 5), dpi=500)
    plt.subplot(R, C, 1)
    plt.plot(df['mse'], df['indep_corr'], 'o-', label='Linear')
    plt.xlabel('Mean-squared error')
    plt.ylabel('corr. between attributes')
    plt.title('Training')

    ax = plt.subplot(R, C, 2)
    plt.plot(df['mse_test'], df['indep_corr_test'], 'o-', label='Linear')
    plt.xlabel('Mean-squared error')
    plt.title('Testing')

    # nonlinear plots
    df2 = pd.read_pickle(oj(PROCESSED_DIR, '06_df_loss_tradeoff_nonlinear.pkl'))
    df2 = df2.iloc[1:]
    df2 = df2.iloc[:-3]

    plt.subplot(R, C, 1)
    plt.plot(df2['mse'], df2['indep_corr'], 'o-', label='Nonlinear')
    plt.ylabel('Mean Inter-Attribute Correlation')

    ax = plt.subplot(R, C, 2)
    plt.plot(df2['mse_test'], df2['indep_corr_test'], 'o-', label='Nonlinear')

    # nonlinear plots
    df2 = pd.read_pickle(oj(PROCESSED_DIR, '07_df_loss_tradeoff_nonlinear_INN.pkl'))
    df2 = df2.iloc[1:]
    df2 = df2.iloc[:-3]

    plt.subplot(R, C, 1)
    plt.plot(df2['mse'], df2['indep_corr'], 'o-', label='Nonlinear-INN')
    plt.ylabel('Mean Inter-Attribute Correlation')

    ax = plt.subplot(R, C, 2)
    plt.plot(df2['mse_test'], df2['indep_corr_test'], 'o-', label='Nonlinear-INN')

    # previous 
    plt.subplot(R, C, 2)

    MSE_ORIG = 0.3671995300361915
    CORR_ORIG = 0.5514424823248604
    plt.plot(MSE_ORIG, CORR_ORIG, 'x', color='black', ms=10)
    ax.annotate('Original setup', (MSE_ORIG + 0.04, CORR_ORIG), color='black')

    MSE_ORTH = 0.51532
    CORR_ORTH = 0.340195386
    plt.plot(MSE_ORTH, CORR_ORTH, 'x', color='black', ms=10)
    ax.annotate('Orthogonal weights', (MSE_ORTH + 0.04, CORR_ORTH), color='black')
    plt.legend()
    plt.tight_layout()
    plt.savefig(oj(RESULTS_DIR, 'fig_attr_mse.pdf'))
    plt.show()