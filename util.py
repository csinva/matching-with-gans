from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

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


def plot_grid(images, ylabs=[], annot_list=None, suptitle=None, emphasize_col: int = None, fontsize_ylab=25):
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
        N_IMS = R * C
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
        fig.text(0.5, 1, suptitle, ha='center')

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
        ax.spines[x].set_linewidth(3)  # ['linewidth'] = 10


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


def wilson(vals, z=1):  # 1.96): # z 1.96 - 95%
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
    center = (ns + z ** 2 / 2) / (n + z ** 2)
    diff = z / (n + z ** 2) * sqrt(ns * nf / n + z ** 2 / 4)
    return center - phat - diff, center - phat + diff
