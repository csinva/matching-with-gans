import sys
import os
import pickle
import numpy as np
import ganwrapper
import imageio
from argparse import ArgumentParser
import pandas as pd

def main(config):

    # Linear models
    model_dir = "./data/latent-models/"

    # Latent codes to use
    seed_path = "./data/annotation-data/W.npy"
    W_all = np.load(seed_path)

    # Output dir
    save_dir_base  = "./results6/"
    save_dir  = save_dir_base + "ims"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    Ws_all = []
    attrs_all = {
        a: [] for a in config.attr
    }
    

    # Set up gpu
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    # Get GAN
    G = ganwrapper.GANWrapper(image_size=512)
    z_dim = G.Gs.input_shape[1]

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

    for i in range( len(config.attr) ):
        idx = all_attrs.index( config.attr[i] )
        plane = planes[idx]
        v = normalize(plane.coef_)
        norm = np.linalg.norm(plane.coef_)

        if config.orth:
            v_orth = orthogonalize(v, planes[0:idx] + planes[idx+1:])
            scale = np.sum(v * v_orth) # Need to change point spacing to account for altered direction.
            dirs.append(v_orth)
        else:
            scale = 1
            dirs.append(v)

        dists = np.linspace(config.lims[i*2], config.lims[i*2+1], config.L[i]) 
        points.append([ (t/norm) / scale for t in dists ])
        grid_planes.append(plane)


    # Create the ND-grid. Need to reverse x,y ordering...I forget exactly why.
    grids = np.meshgrid( *(points[0:2][::-1] + points[2:])  )
    dirs  = dirs[0:2][::-1] + dirs[2:]

    # Offset vectors from boundary in latent space
    deltas = sum( [ np.expand_dims(dirs[i],0) * np.expand_dims(grids[i],-1) 
                    for i in range(len(grids)) ] )
    deltas = np.reshape(deltas, (-1, z_dim), 'F')

    batch_size = 1 
    n_batch = config.N//batch_size

    # Transect creation loop.
    for ex_num in range(config.N):
        print(ex_num, n_batch, flush=True)
   
        # Project onto intersection of attribute hyperplanes
        W0 = projectToBoundary(W_all[ex_num, ...], grid_planes)

        for j, delta in enumerate(deltas):
            W = W0 + delta
            Ws_all.append(W)
            
            idx = np.unravel_index(j, config.L, 'F')
            
            # save the attributes
            for a,b in zip(config.attr, idx):
                attrs_all[a].append(int(b))
                
            # generate and save the images
            img = G.generateImageFromStyle(W)
            img = (img*255).astype(np.uint8)

            # Filename: version + seed face + attrs
            fname = '%s/%.2d_%.4d' % (save_dir, config.version, ex_num)
            

            for a,b in zip(config.attr, idx):
                fname += '_' + a + str(int(b))

            imageio.imwrite(fname + '.jpg', img[0, ...])           
            
            
    # save
    pd.DataFrame.from_dict(attrs_all).to_csv(save_dir_base + 'attrs.csv')
    np.save(save_dir_base + 'Ws.npy', np.array(Ws_all))


def projectToBoundary(X, planes, n_iter=100):
    n_planes = len(planes)

    if n_planes == 1:
        return projectToPlane(X, planes[0].coef_, planes[0].intercept_)

    # Iterative procedure for getting to boundary
    for i in range(n_iter):
        plane = planes[i % n_planes]
        X = projectToPlane(X, plane.coef_, plane.intercept_)

    return X   


def projectToPlane(X, w, b):
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
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--attr', type=str, nargs='+')
    parser.add_argument('--L', type=int, nargs='+')
    parser.add_argument('--lims', type=float, nargs='+')
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--orth', type=int, default=1)
    parser.add_argument('--N', type=int)

    config = parser.parse_args()

    np.random.seed(2)
    main(config)
