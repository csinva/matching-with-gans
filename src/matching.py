import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics


def get_lat(latents):
    #     lat = latents.mean(axis=1) # match in style space
    lat = latents.reshape(latents.shape[0], -1)  # match in extended style space
    return lat


def get_dists(lat):
    '''Get distances
    '''
    # calculate distances: (points, points)
    #     dists = sklearn.metrics.pairwise_distances(lat, metric='cosine') # cosine dist
    dists = sklearn.metrics.pairwise_distances(lat, metric='l2')  # l2 dist
    # sns.clustermap(dists)
    dists[np.eye(dists.shape[0]).astype(bool)] = 1e3  # don't pick same point
    # plt.imshow(dists)
    # dists_min = np.argmin(dists, axis=1)
    return dists


def join_vecs(preds, lats, weights, discretize=False):
    '''Joins preds and lat with equal weight, optionally weighting preds
    preds
        (N_images, N_attr)
    weights: np.ndarray
        length (N_attr), containing weight for each attr of the predictions
    '''
    lats_norm = (lats - lats.mean(axis=0)) / (lats.std(axis=0) + 1e-10)
    preds_norm = (preds - preds.mean(axis=0)) / (preds.std(axis=0) + 1e-10)
    if discretize:
        preds_norm = (preds_norm > 0).astype(int)
    preds_norm = (preds - preds.mean(axis=0)) / (preds.std(axis=0) + 1e-10)
    preds_norm = preds_norm * weights

    vecs = np.hstack((preds_norm, lats_norm))
    return vecs


def calc_matches(dists, fname_ids):
    '''
    dists: np.array
        dists for current image
    fname_ids: np.ndarray
        fnames ids corresponding to dists
    '''
    closest_matches = np.argsort(dists)  # dists[im_num - 1])
    return dists[closest_matches], fname_ids[closest_matches]


def show_matches(dists, DIR_ORIG, DIR_GEN, fname_ids, im_nums=range(60, 70)):
    # pick the image
    for im_num in im_nums:
        R, C = 1, 7
        plt.figure(figsize=(C * 2, R * 2))

        plt.subplot(R, C, 1)
        im_orig = mpimg.imread(oj(DIR_ORIG, f'{im_num:05}.jpg'))
        util.imshow(im_orig)
        plt.title('original im', fontsize=10)

        plt.subplot(R, C, 2)
        im_rec = mpimg.imread(oj(DIR_GEN, f'{im_num:05}.png'))
        util.imshow(im_rec)
        plt.title('reconstruction', fontsize=10)

        #     print(dists[im_num - 1][closest_matches])
        plt.subplot(R, C, 3)
        plt.title('closest matches...', fontsize=10)
        for i in range(C - 2):
            plt.subplot(R, C, i + 3)
            matched_num = closest_matches[i] + 1
            im = mpimg.imread(oj(DIR_GEN, f'{matched_num:05}.png'))
            util.imshow(im)
        plt.show()
