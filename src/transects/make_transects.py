import sys
import os
import pickle
import numpy as np
import ganwrapper
import imageio
from argparse import ArgumentParser
import pandas as pd
from os.path import join as oj

def make_transects(G,
                   attr: list=['A', 'C', 'H', 'G'],
                   N_IMS_LIST: list=[1, 2, 2, 2],
                   LIMS_LIST: list=[0.5, 0.5, -1.5, 1.7, -0.5, 0.0, -1.75, 1.75],
                   N: int=10,
                   orth: bool=False,
                   save_dir: str='results',
                   randomize_seeds: bool=False,
                   model_dir: str="./data/latent-models/",
                   latents: np.ndarray=None,
                   seed_path: str="./data/annotation-data/W.npy"):
    '''
    Params
    ------
    N
        Number of transects to generate
    N_IMS_LIST
        List of number of images to generate in each direction
        
    latents
        Numpy of array to start making latents
        Either (N, 512) or (N, 18, 512)
    randomize_seeds
        If this is true and latents not passed, generate random latents
    seed_path
        If latents is None and randomize_seeds is False, then
        generate using seeds at this path
        
    '''
    
    # Output dir
    save_dir_base  = save_dir
    save_dir  = oj(save_dir_base, "ims")
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
        W_all = latents
    elif randomize_seeds:
        W_all = G.getStyle(np.random.randn(N, 512))    
    else:
        W_all = np.load(seed_path)
    N = min(N, W_all.shape[0])

    # Attributes coded by letters
    all_attrs = 'HAGCBMSEW'

    # Load models
    planes = []
    for a in all_attrs:
        filename = os.path.join(model_dir, a + "_W.pkl")
        with open(filename, 'rb') as pickle_file:
            data = pickle.load(pickle_file)

            # LinearSVC and RidgeCV save hyperplanes slightly differently
            if len(data.coef_.shape) == 1:
                data.coef_ = np.expand_dims(data.coef_, 0)
                data.intercept_ = np.expand_dims(data.intercept_, 0)

            planes.append(data)

    # Get attribute normal vectors and step sizes
    dirs = []  # Direction vectors
    points = [] # Distances from hyperplanes to query
    grid_planes = [] # Hyperplanes for grid attributes

    for i in range( len(attr) ):
        idx = all_attrs.index( attr[i] )
        plane = planes[idx]
        v = normalize(plane.coef_)
        norm = np.linalg.norm(plane.coef_)

        if orth:
            v_orth = orthogonalize(v, planes[0:idx] + planes[idx+1:])
            scale = np.sum(v * v_orth) # Need to change point spacing to account for altered direction.
            dirs.append(v_orth)
        else:
            scale = 1
            dirs.append(v)

        dists = np.linspace(LIMS_LIST[i * 2], LIMS_LIST[i * 2 + 1], N_IMS_LIST[i]) 
        points.append([ (t/norm) / scale for t in dists ])
        grid_planes.append(plane)


    # Create the ND-grid. Need to reverse x,y ordering...I forget exactly why.
    grids = np.meshgrid( *(points[0:2][::-1] + points[2:])  )
    dirs  = dirs[0:2][::-1] + dirs[2:]

    # Offset vectors from boundary in latent space
    deltas = sum( [ np.expand_dims(dirs[i],0) * np.expand_dims(grids[i],-1) 
                    for i in range(len(grids)) ] )
    z_dim = G.Gs.input_shape[1]
    deltas = np.reshape(deltas, (-1, z_dim), 'F')

    batch_size = 1 
    n_batch = N//batch_size

    # Transect creation loop.
    for ex_num in range(N):
        print(ex_num, n_batch, flush=True)
   
        # Project onto intersection of attribute hyperplanes
        if len(W_all.shape) == 3: # We have a real image, (N, 18, 512)
            W0 = W_all[ex_num: ex_num + 1]
        else:
            W_seed = W_all[ex_num, ...]
            W0 = projectToBoundary(W_seed, grid_planes)
#         print('W0.shape', W0.shape)
        for j, delta in enumerate(deltas):
            W = W0 + delta
#             W = W0
            Ws_all.append(W)
            
            idx = np.unravel_index(j, N_IMS_LIST, 'F')
            
            # save the attributes
            for a,b in zip(attr, idx):
                attrs_all[a].append(int(b))
                
            # generate and save the images
            if len(W.shape) == 3:
                img = G.generateImageFromStyleFull(W)
            else:
                img = G.generateImageFromStyle(W)
            img = (img * 255).astype(np.uint8)

            # Filename: seed face + attrs
            fname = f'{save_dir}/{ex_num:04d}'
            for a,b in zip(attr, idx):
                fname += '_' + a + str(int(b))
            attrs_all['fnames'].append(fname)
            imageio.imwrite(fname + '.jpg', img[0, ...])           
            
            
    # save
    pd.DataFrame.from_dict(attrs_all).to_csv(oj(save_dir_base, 'attrs.csv'))
    np.save(oj(save_dir_base, 'Ws.npy'), np.array(Ws_all))


def projectToBoundary(X, planes, n_iter=100):
    '''
    Params
    ------
    X: np.ndarray (1, 512)
        point to be projected
    planes: sklearn linear models with .coef_ and .intercept)
        ws: np.ndarray (512, n_attributes)
            coefficients for different linear models
        bs: array_like (n_attributes)
            intercepts for linear model
    '''    
    n_planes = len(planes)

    if n_planes == 1:
        return projectToPlane(X, planes[0].coef_, planes[0].intercept_)

    # Iterative procedure for getting to boundary
    for i in range(n_iter):
        plane = planes[i % n_planes]
        X = projectToPlane(X, plane.coef_, plane.intercept_)

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
    b = b/np.linalg.norm(w)
    w = normalize(w)

    # Project points back to hyperplane, decision value = 0
    d = np.sum(w * X, -1) + b

    return X - w * np.expand_dims(d, 1)


def orthogonalize(v0, planes):

    A = np.zeros((max(v0.shape), len(planes)))
    for i,h in enumerate(planes):
        A[:, i] = h.coef_
    Q,R = np.linalg.qr(A)

    u = v0.copy()
    for i in range(len(planes)):
        u -= proj(Q[:,i], u)

    return normalize(u)
    

def normalize(u):
    return u/np.linalg.norm(u)


def proj(u, v):
    return u * np.sum(u * v) / np.sum( u * u)


if __name__ == "__main__":

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
    parser.add_argument('--version', type=int, default=1) # not used
    config = parser.parse_args()
    print('config', config, '\n\n')
    
    # Get GAN
    np.random.seed(2)
    G = ganwrapper.GANWrapper(image_size=512)
    make_transects(G,
         N_IMS_LIST=config.L,
         attr=config.attr,
         N=config.N,
         orth=config.orth,
         LIMS_LIST=config.lims,
         save_dir=config.save_dir,
         randomize_seeds=config.randomize_seeds
        )
