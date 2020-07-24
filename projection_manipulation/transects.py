import os
import pickle
import sys
from argparse import ArgumentParser
from os.path import join as oj

import imageio
import numpy as np
import pandas as pd

from ganwrapper import Generator

sys.path.append('..')
import config
from copy import deepcopy
from tqdm import tqdm


def get_directions(model_dir=config.DIR_LINEAR_DIRECTIONS, all_attrs=config.ALL_ATTRS):
    '''
    Returns
    -------
    coefs: list
        (N_attributes, 1, 512)
    intercepts: list
        (N_attributes, 1)
    '''

    coefs = []
    intercepts = []
    if model_dir is not None:
        for a in all_attrs:
            filename = os.path.join(model_dir, a + "_W.pkl")
            with open(filename, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                # LinearSVC and RidgeCV save hyperplanes slightly differently
                if len(data.coef_.shape) == 1:
                    data.coef_ = np.expand_dims(data.coef_, 0)
                    data.intercept_ = np.expand_dims(data.intercept_, 0)
                coefs.append(data.coef_)
                intercepts.append(data.intercept_)
    return coefs, intercepts


def make_transects(G,
                   attr: list = ['A', 'C', 'H', 'G'],
                   N_IMS_LIST: list = [1, 2, 2, 2],
                   LIMS_LIST: list = [0.5, 0.5, -1.5, 1.7, -0.5, 0.0, -1.75, 1.75],
                   N: int = 10,
                   orth: bool = False,
                   save_dir: str = 'results',
                   randomize_seeds: bool = False,
                   model_dir: str = config.DIR_LINEAR_DIRECTIONS,
                   latents: np.ndarray = None,
                   seed_path: str = "./linear_models/annotation-data/W.npy",
                   return_ims: bool = False,
                   force_project_to_boundary: bool = False,
                   return_project_to_boundary: bool = False):
    '''
    Params
    ------
    N
        Number of transects to generate
    N_IMS_LIST
        List of number of images to generate in each direction
        
    latents: array_like
        Array to start making latents
        Either (N, 512) or (N, 18, 512)
    randomize_seeds
        If this is true and latents not passed, generate random latents
    seed_path
        If latents is None and randomize_seeds is False, then
        generate using seeds at this path
    force_project_to_boundary
        Whether to force project something in the expanded latent space
        
    '''

    # Output dir
    save_dir_base = save_dir
    save_dir = oj(save_dir_base, "ims")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Ws_all = []
    attrs_all = {
        a: [] for a in attr
    }
    attrs_all['fnames'] = []

    # Set up gpu
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    # Latent codes to use (should end up (N, 512) or (N, 18, 512))
    # If it is (N, 18, 512), will manipulate it along each dimension
    if latents is not None:
        latents = np.array(latents)
        W_all = latents
    elif randomize_seeds:
        W_all = G.getStyle(np.random.randn(N, 512))
    else:
        W_all = np.load(seed_path)
    N = min(N, W_all.shape[0])

    # Attributes coded by letters
    all_attrs = config.ALL_ATTRS

    # Load models
    coefs, intercepts = get_directions(model_dir, all_attrs)

    # Get attribute normal vectors and step sizes
    dirs = []  # Direction vectors
    points = []  # Distances from hyperplanes to query
    grid_coefs = []  # Hyperplanes for grid attributes
    grid_intercepts = []  # Hyperplanes for grid attributes

    for i in range(len(attr)):
        idx = all_attrs.index(attr[i])
        v = normalize(coefs[idx])
        norm = np.linalg.norm(coefs[idx])

        if orth:
            coefs_other = coefs[0:idx] + coefs[idx + 1:]
            intercepts_other = intercepts[0:idx] + intercepts[idx + 1:]

            v_orth = orthogonalize(v, coefs_other)
            scale = np.sum(v * v_orth)  # Need to change point spacing to account for altered direction.
            dirs.append(v_orth)
        else:
            scale = 1
            dirs.append(v)

        dists = np.linspace(LIMS_LIST[i * 2], LIMS_LIST[i * 2 + 1], N_IMS_LIST[i])
        points.append([(t / norm) / scale for t in dists])
        grid_coefs.append(coefs[idx])
        grid_intercepts.append(intercepts[idx])

    # Create the ND-grid. Need to reverse x,y ordering...I forget exactly why.
    grids = np.meshgrid(*(points[0:2][::-1] + points[2:]))
    dirs = dirs[0:2][::-1] + dirs[2:]

    # Offset vectors from boundary in latent space
    deltas = sum([np.expand_dims(dirs[i], 0) * np.expand_dims(grids[i], -1)
                  for i in range(len(grids))])
    z_dim = G.Gs.input_shape[1]
    deltas = np.reshape(deltas, (-1, z_dim), 'F')


    # Transect creation loop
    ims = []
    for ex_num in tqdm(range(N)):

        # Project onto intersection of attribute hyperplanes
        if len(W_all.shape) == 3:  # W_all contains latents in expanded space (N, 18, 512)
            W0 = W_all[ex_num: ex_num + 1]
            if force_project_to_boundary:
                W_mean = np.mean(W0, axis=1)
                W0_proj = projectToBoundary(W_mean, grid_coefs, grid_intercepts)  # grid_planes)
                W0 = W0 - W_mean  + W0_proj
        else:  # W_all contains latents in original space (N, 512)
            W_seed = W_all[ex_num, ...]
            W0 = projectToBoundary(W_seed, grid_coefs, grid_intercepts)  # grid_planes)
        for j, delta in enumerate(deltas):
            W = W0 + delta
            Ws_all.append(W)

            idx = np.unravel_index(j, N_IMS_LIST, 'F')

            # save the attributes
            for a, b in zip(attr, idx):
                attrs_all[a].append(int(b))

            # generate and save the images
            if len(W.shape) == 3:
                img = G.generateImageFromStyleFull(W)
                if return_project_to_boundary:
                    return img, W
            else:
                img = G.generateImageFromStyle(W)
            img = (img * 255).astype(np.uint8)

            # Filename: seed face + attrs
            fname = f'{save_dir}/{ex_num:04d}'
            for a, b in zip(attr, idx):
                fname += '_' + a + str(int(b))
            attrs_all['fnames'].append(fname)
            imageio.imwrite(fname + '.jpg', img[0, ...])
            if return_ims:
                ims.append(deepcopy(img[0]))

    # save
    attr_df = pd.DataFrame.from_dict(attrs_all)
    attr_df.to_csv(oj(save_dir_base, 'attrs.csv'))
    np.save(oj(save_dir_base, 'Ws.npy'), np.array(Ws_all))

    if return_ims:
        return ims, attr_df


def projectToBoundary(X, coefs, intercepts, n_iter=100):
    '''
    Params
    ------
    X: np.ndarray (1, 512)
        point to be projected
    coefs: array_like (n_attributes, 512)
        coefficients for different linear models
    intercepts: array_like (n_attributes, 1)
        intercepts for linear model
    '''
    try:
        n_planes = len(coefs)
    except:
        n_planes = coefs.shape[0]

    if n_planes == 1:
        return projectToPlane(X, coefs[0], intercepts[0])

    # Iterative procedure for getting to boundary
    for i in range(n_iter):
        idx = i % n_planes
        # plane = planes[]
        X = projectToPlane(X, coefs[idx], intercepts[idx])

    return X


def projectToPlane(X, w, b):
    '''
    Params
    ------
    X: np.ndarray (1, 512)
        point to be projected
    w: np.ndarray (512)
        coefficients for linear model
    b: scalar
        intercept for linear model
    '''
    w, b = w.copy(), b.copy()

    # Get and normalize coefficients
    b = b / np.linalg.norm(w)
    w = normalize(w)

    # Project points back to hyperplane, decision value = 0
    d = np.sum(w * X, -1) + b

    return X - w * np.expand_dims(d, 1)


def orthogonalize(v0, coefs):
    A = np.zeros((max(v0.shape), len(coefs)))
    for i, coef in enumerate(coefs):
        A[:, i] = coef  # h.coef_
    Q, R = np.linalg.qr(A)

    u = v0.copy()
    for i in range(len(coefs)):
        u -= proj(Q[:, i], u)

    return normalize(u)


def normalize(u):
    return u / np.linalg.norm(u)


def proj(u, v):
    return u * np.sum(u * v) / np.sum(u * u)


if __name__ == "__main__":
    # example
    # python make_transects.py --gpu 1 --attr A C H G --L 1 2 2 2 --lims 0.5 0.5 -1.5 1.7 -0.5 0 -1.75 1.75 --version 1 --orth 1 --N 1000
    # python make_transects.py --gpu 1 --attr C H G --L 2 2 2 --lims -1.5 1.7 -0.5 0 -1.75 1.75 --version 1 --orth 1 --N 10
    # python make_transects.py --gpu 1 --attr H --L 7 --lims -0.5 0 --version 1 --orth 1 --N 10
    # python make_transects.py --gpu 1 --attr C --L 7 --lims -1.5 1.7 --version 1 --orth 1 --N 10
    # python make_transects.py --attr A C H G --L 1 2 2 2 --lims 0.5 0.5 -1.5 1.7 -0.5 0 -1.75 1.75 --orth 1 --N 10 --save_dir 'test' --randomize_seeds
    
    
    parser = ArgumentParser()
    parser.add_argument('--attr', type=str, nargs='+')
    parser.add_argument('--L', type=int, nargs='+')
    parser.add_argument('--lims', type=float, nargs='+')
    parser.add_argument('--orth', type=int, default=1)
    parser.add_argument('--N', type=int)
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--randomize_seeds', default=False, action='store_true')

    # not used
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--version', type=int, default=1)  # not used
    config = parser.parse_args()
    print('config', config, '\n\n')

    # Get GAN
    np.random.seed(2)
    G = Generator(image_size=512)
    make_transects(G,
                   N_IMS_LIST=config.L,
                   attr=config.attr,
                   N=config.N,
                   orth=config.orth,
                   LIMS_LIST=config.lims,
                   save_dir=config.save_dir,
                   randomize_seeds=config.randomize_seeds
                   )
