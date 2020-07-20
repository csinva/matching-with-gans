import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import scipy.stats
import torch
from os.path import join as oj
import os
from math import sqrt
import style
from config import *

def savefig(fname):
    if '.' in fname:
        print('filename should not contain extension!')
    if not fname.startswith('fig_'):
        fname = 'fig_' + fname
    os.makedirs(DIR_FIGS, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(oj(DIR_FIGS, fname) + '.pdf', bbox_inches='tight')
    plt.savefig(oj(DIR_FIGS, fname) + '.png', dpi=300, bbox_inches='tight')

def corrplot(corrs):
    mask = np.triu(np.ones_like(corrs, dtype=np.bool))
    corrs[mask] = np.nan
    max_abs = np.nanmax(np.abs(corrs))
    plt.imshow(corrs, cmap=style.cmap_div, vmax=max_abs, vmin=-max_abs)


def plot_row(images, annot_list: list = None, dpi: int = 100,
             suptitle: str = None, ylab: str = None, fontsize_ylab=25):
    '''
    Params
    ------
    images: np.ndarray
        (num_images, H, W, C)
    '''

    # deal with inputs
    if type(images) == list:
        N_IMS = len(images)
    else:
        N_IMS = images.shape[0]
    if annot_list is None:
        annot_list = [None] * N_IMS

    fig = plt.figure(figsize=(N_IMS * 3, 3), dpi=dpi)
    for i in range(N_IMS):
        ax = plt.subplot(1, N_IMS, i + 1)
        imshow(images[i], annot=annot_list[i])
        if i == 0:
            show_ylab(ax, ylab, fontsize_ylab=fontsize_ylab)
#             plt.ylabel(ylab, fontsize=fontsize_ylab)
#             fig.text(0, 0.5, ylab, rotation=90, va='center', fontsize=fontsize_ylab)
    if suptitle is not None:
        plt.subplot(1, N_IMS, N_IMS // 2 + 1)
        plt.title(suptitle)
#     if ylab is not None:
        
    plt.tight_layout()


def plot_grid(images, ylabs=[], annot_list=None, suptitle=None, emphasize_col: int=None, fontsize_ylab=25):
    '''
    Params
    ------
    images: np.ndarray
        (R, C, H, W, C)
    emphasize_col
        which column to emphasize (by not removing black border)
    '''
    
    # deal with inputs
    if type(images) == list:
        images = np.array(images)
    #     print(images.shape)
    # check if wasn't passed a grid
    if len(images.shape) == 4:
        N_IMS = images.shape[0]
        R = int(np.sqrt(N_IMS))
        C = R + 1
    else:
        R = images.shape[0]
        C = images.shape[1]
        # reshape to be (R * C, H, W, C)
        images = images.reshape((R * C, *images.shape[2:]))
    if annot_list is None:
        annot_list = [None] * N_IMS
        
    i = 0
    fig = plt.figure(figsize=(C * 3, R * 3))
    for r in range(R):
        for c in range(C):
            ax = plt.subplot(R, C, i + 1)
            imshow(images[r * C + c], annot=annot_list[i])

            if c == 0 and len(ylabs) > r:
                show_ylab(ax, ylabs[r], fontsize_ylab=fontsize_ylab)

            i += 1
            if i >= images.shape[0]:
                break
                
            if c == emphasize_col:
                emphasize_box(ax)

    if suptitle is not None:
        fig.text(0.5, 1, suptitle,  ha='center')

    '''
    if ylabs is not None:
        for r in range(R):
            fig.text(0, r / R + 0.5 / R, ylabs[R - 1 - r], rotation=90,
                         va='center', fontsize=fontsize_ylab)
    ''' 
    fig.tight_layout()
    
def show_ylab(ax, ylab, fontsize_ylab):
    plt.axis('on')
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for x in ['right', 'top', 'bottom', 'left']:
        ax.spines[x].set_visible(False)
    plt.ylabel(ylab, fontsize=fontsize_ylab)

def emphasize_box(ax):
    plt.axis('on')
    ax.get_yaxis().set_ticks([])
    ax.get_xaxis().set_ticks([])
    for x in ['right', 'top', 'bottom', 'left']:
        ax.spines[x].set_visible(True)
        ax.spines[x].set_linewidth(3) #['linewidth'] = 10
#         [i.set_linewidth(0.1) for i in ax.spines.itervalues()]
#     ax.spines['top'].set_visible(True)


def norm(im):
    '''Normalize to [0, 1]
    '''
    return (im - np.min(im)) / (np.max(im) - np.min(im))  # converts range to [0, 1]


def imshow(im, annot: str = None):
    '''
    Params
    ------
    annot
        str to put in top-right corner
    '''

    # if 4d, take first image
    if len(im.shape) > 3:
        im = im[0]

    # if channels dimension first, transpose
    if im.shape[0] == 3 and len(im.shape) == 3:
        im = im.transpose()

    ax = plt.gca()
    ax.imshow(im)
    ax.axis('off')

    if annot is not None:
        padding = 5
        ax.annotate(
            s=annot,
            fontsize=12,
            xy=(0, 0),
            xytext=(padding - 1, -(padding - 1)),
            textcoords='offset pixels',
            bbox=dict(facecolor='white', alpha=1, pad=padding),
            va='top',
            ha='left')


def detach(tensor):
    return tensor.detach().cpu().numpy()


def orthogonalize_paper(vs: np.ndarray):
    '''
    Params
    ------
    vs
        matrix of all vectors to orthogonalize
        (each col is a vector)
    '''
    d = vs.shape[1]  # number of vectors
    Q, R = npl.qr(vs)
    vs_orth = vs.copy()
    for i in range(d):
        for j in range(d):
            if not i == j:
                scalar = np.dot(Q[:, j], vs_orth[:, i]) / npl.norm(Q[:, j])
                vs_orth[:, i] -= Q[:, j] * scalar
        vs_orth[:, i] = vs_orth[:, i] / npl.norm(vs_orth[:, i])
    return vs_orth


def spearman_mean(y_pred: torch.Tensor, y_train: torch.Tensor):
    '''
    Params
    ------
    y_pred
        (n_samples, n_attributes)
    y_train
        (n_samples, n_attributes)
        
    Returns
    -------
    mean_rho: float
        mean spearman correlation between corresponding columns
        of y_pred and y_train
    '''

    spearman_cum = 0
    for i in range(y_pred.shape[1]):
        spearman_cum += scipy.stats.spearmanr(detach(y_pred[:, i]),
                                              detach(y_train[:, i])).correlation
    return spearman_cum / y_pred.shape[1]


def pearsonr(x, y):
    """
    Mimics `scipy.stats.pearsonr`

    Arguments
    ---------
    x : 1D torch.Tensor
    y : 1D torch.Tensor

    Returns
    -------
    r_val : float
        pearsonr correlation coefficient between x and y
    
    Scipy docs ref:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    
    Scipy code ref:
        https://github.com/scipy/scipy/blob/v0.19.0/scipy/stats/stats.py#L2975-L3033
    Example:
        >>> x = np.random.randn(100)
        >>> y = np.random.randn(100)
        >>> sp_corr = scipy.stats.pearsonr(x, y)[0]
        >>> th_corr = pearsonr(torch.from_numpy(x), torch.from_numpy(y))
        >>> np.allclose(sp_corr, th_corr)
    """
    xm = x - torch.mean(x)
    ym = y - torch.mean(y)
    return xm.dot(ym) / (torch.norm(xm, 2) * torch.norm(ym, 2))


def orthogonalize(v0: np.ndarray, vs: np.ndarray):
    '''
    Params
    ------
    v0: (size)
        vector to orthogonalize
    vs: (num_vectors x size)
        vectors to orthogonalize against
    '''
    '''
    # convert other vectors to a matrix
    A = np.zeros((max(v0.shape), len(vs)))
    for i, h in enumerate(vs):
        A[:, i] = h.coef_
    '''

    # decompose the matrix
    vs = vs.transpose()  # make it (size x num_vectors)
    Q, R = npl.qr(vs)

    # subtract projections onto other vectors
    u = v0.copy()
    for i in range(vs.shape[1]):
        u -= proj(Q[:, i], u)

    # normalize
    u = u / npl.norm(u)
    return u


def proj(u: np.ndarray, v: np.ndarray):
    '''Return projection of u onto v
    '''
    return u * np.dot(u, v) / np.dot(u, u)


def wilson(vals, z=1): #1.96): # z 1.96 - 95%
    '''vals is array_like of binary values
    
    Returns
    -------
    (err_lower, err_upper) around the mean
    '''
    vals = np.array(vals)
    nf = np.sum(vals == 0)
    ns = np.sum(vals)
    n = vals.size

    # implemented based on eq. from wikipedia as of jul 17 2020
    phat = float(ns) / n
    center = (ns + z**2 / 2) / (n + z**2)
    diff = z / (n + z**2) * sqrt(ns * nf / n + z**2 / 4)
    return center - phat - diff, center - phat + diff