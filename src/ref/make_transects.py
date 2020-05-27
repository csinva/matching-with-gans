import sys
import os
import pickle
import numpy as np
import ganwrapper
import imageio
import itertools
from argparse import ArgumentParser


def main(config):

    base_dir  = "/data/vision/billf/scratch/balakg/amn"
    model_dir = "./results/latent-models/" #"%s/latent-models/v%d" % (base_dir, config.gan)
    seed_path = "./results/annotation-data/W.npy" #"%s/annotation-data-%d/W.npy" % (base_dir, config.gan)
    save_dir  = "./results/outputs/" #"%s/transects/v%d/%s" % (base_dir, config.gan, '-'.join(config.attr))

    #model_dir = "%s/latent-models/v%d" % (base_dir, config.gan)
    #seed_path = "%s/annotation-data-%d/W.npy" % (base_dir, config.gan)
    #save_dir  = "%s/transects/v%d/%s" % (base_dir, config.gan, '-'.join(config.attr))

    batch     = 1 
    all_attrs = 'HAGCBMSEW'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get GAN
    G = ganwrapper.GANWrapper(version = config.gan, image_size=512)
    z_dim = G.Gs.input_shape[1]

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


    # Create the ND-grid. Need to reverse x,y ordering...something to do with saving
    # the images with the right ordering.
    grids = np.meshgrid( *(points[0:2][::-1] + points[2:])  )
    dirs  = dirs[0:2][::-1] + dirs[2:]

    # Offset vectors from boundary in latent space
    deltas = sum( [ np.expand_dims(dirs[i],0) * np.expand_dims(grids[i],-1) 
                    for i in range(len(grids)) ] )
    deltas = np.reshape(deltas, (-1, z_dim), 'F')

    n_batch = config.N//batch

    # Transect creation loop.
    for i in range(config.N):
        print(i, n_batch, flush=True)
   
        # Draw random z
        z = np.random.randn(1, z_dim)
        W_i = G.getStyle(z)

        # Project onto intersection of attribute hyperplanes
        W0 = projectToBoundary(W_i, grid_planes) #W_all[i], grid_planes)

        for j, delta in enumerate(deltas):
            W = W0 + delta

            img = G.generateImageFromStyle(W)
            img = (img*255).astype(np.uint8)

            for k in range(batch):
                
                # Save image.
                ex_num = i #* batch + k 

                # Filename: version + seed face + attrs
                fname = '%s/%.2d_%.4d' % (save_dir, config.version, ex_num)
                idx = np.unravel_index(j, config.L, 'F')

                for a,b in zip(config.attr, idx):
                    fname += '_' + a + str(int(b))

                fname += '.jpg' 
                imageio.imwrite(fname, img[k, ...]) 

                #print(p, fname, flush=True)


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
    parser.add_argument('--attr', type=str, nargs='+', help='Attributes (specified in letters)')
    parser.add_argument('--L', type=int, nargs='+', help='transect dimensions')
    parser.add_argument('--lims', type=float, nargs='+')
    parser.add_argument('--gan', type=int, default=1)
    parser.add_argument('--version', type=int, default=1, help='Version number for saving files')
    parser.add_argument('--orth', type=int, default=1, help='1 to use orthogonalization, 0 otherwise')
    parser.add_argument('--N', type=int, help='number of transects')

    config = parser.parse_args()

    #np.random.seed(2) #2
    main(config)
