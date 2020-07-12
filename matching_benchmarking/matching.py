from os.path import join as oj

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import util


def get_lat(latents):
    #     lat = latents.mean(axis=1) # match in style space
    lat = latents.reshape(latents.shape[0], -1)  # match in extended style space
    return lat


def calc_dists_from_latents(lat):
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


# specify constraint for reference
def apply_reference_constraints(d):
    REFERENCE_CONSTRAINTS = {
        'eyeglasses': 0, # default should be no eyeglasses
    }
    for k in REFERENCE_CONSTRAINTS.keys():
        d = d[d[k] == REFERENCE_CONSTRAINTS[k]]
    return d

## select only ims with a reference image satisfying reference constraints
def get_idxs_satisfying_reference_constraints(df, dists_match, dists_ref):
    n = dists_match.shape[0]
    idxs_orig = pd.Series(1, df.index) #df['bool'] # np.ones(n).astype(bool)
    
    # prune anything for which dists are constant (the code for NaN)
    print('pruning anything with constant dists...')
    for i in tqdm(range(n)):
        if np.all(dists_match[i] == dists_match[i, 0]) or np.all(dists_ref[i] == dists_ref[i, 0]):
            idxs_orig.iloc[i] = False
    
    # prune things based on reference constraints
    print('pruning based on reference constraints...')
    for i in tqdm(sorted(df.id.unique())):
        d = df[(df.id == i) & idxs_orig]

        # if there is only one photo for this id, don't pick it
        if d.shape[0] == 1:
            idxs_orig.loc[d.index] = False
        
        else:
            # look for valid ref photos
            d_ref = apply_reference_constraints(d)

            # if there is no valid ref, don't pick any image with this id
            if d_ref.shape[0] < 1:
                idxs_orig.loc[d.index] = False

            # if there is 1 valid ref, don't pick the reference image
            # (we can still pick a different photo with this id)
            elif d_ref.shape[0] == 1:
                idxs_orig.loc[d_ref.index] = False 
                
    return idxs_orig.values.astype(bool)


def get_matches(df, dists_match, dists_ref, attrs_to_vary,
                NUM_MATCHES, MIN_REF_DIST_THRESH_UPPER, MIN_REF_DIST_THRESH_LOWER):
    '''Run full matching
    
    attrs_to_vary: List[str]
        assumes that each attr is binary (0, 1) with other values ignored
    
    '''
    
    # first prune to images satisfying constraints
    n = df.shape[0]
    idxs_orig = get_idxs_satisfying_reference_constraints(df, dists_match, dists_ref)
    print('total ims', n, 'selectable ims', np.sum(idxs_orig))
    print('total ids', df.id.unique().size, 'selectable ids', df[idxs_orig].id.unique().size)
    
    
    # find allowed indices for each group
    subgroups = {}
    for a in attrs_to_vary:
        for val in [0, 1]:
            subgroups[f'{a}_{val}'] = (df[a].values == val) & idxs_orig
            
    # check passed args
    assert n == dists_match.shape[0], 'df shapes must match'
    assert n == dists_ref.shape[0], 'ref shapes must match'
    
    
    # start calculating each individual match
    matches = {}
    matches_skipped = []
    pairwise_constraints = np.zeros((n, n)).astype(bool) # extra constraints for matching
    print(f'computing {NUM_MATCHES} matches...')
    for match_num in tqdm(range(NUM_MATCHES)):
        # loop to create best matches
        for i, a in enumerate(attrs_to_vary):
            # this should go in order from smaller groups -> bigger groups
        #     vals = np.argsort([np.sum(subgroups[f'{a}_{val}']) for val in [0, 1]])
        #     for j, val in enumerate([0, 1]):

            ## find best match subject to constraints
            s0 = f'{a}_{0}'
            s1 = f'{a}_{1}'
            idxs0 = subgroups[s0]
            idxs1 = subgroups[s1]
            C = np.sum(idxs1)

            # if there are no more possible matches, stop
            if np.sum(idxs0) == 0 or np.sum(idxs1) == 0:
                NUM_MATCHES = len(matches[s0])
                break

            # extra constraints (e.g. previously skipped, could enforce that attributes are the same)
            dists_match_constrained = deepcopy(dists_match)
            dists_match_constrained[pairwise_constraints] = 1e5

            # main constraints (e.g. attributes are different, previously selected)
            dists_match_constrained = dists_match_constrained[idxs0][:, idxs1] # (R, C)

            arg = np.argmin(dists_match_constrained)

            # convert match arg back to space without constraints
            arg0 = arg // C
            arg1 = arg % C
            idx0 = np.where(idxs0)[0][arg0]
            idx1 = np.where(idxs1)[0][arg1]

            ## find reference images
            idxs_ref = pd.Series(True, df.index)
            id0 = df.iloc[idx0].id
            id1 = df.iloc[idx1].id
            idxs_ref0 = idxs_ref & (df.id == id0) & (np.arange(n) != idx0)
            idxs_ref1 = idxs_ref & (df.id == id1) & (np.arange(n) != idx1)
            assert np.sum(idxs_ref0) > 0 and np.sum(idxs_ref1) > 0, 'must have valid reference images'

            ## compare dists for reference images
            dists0 = dists_ref[idx0][idxs_ref0].reshape(-1, 1)
            dists1 = dists_ref[idx1][idxs_ref1].reshape(-1, 1)

            ## pairwise distances
            dists_ref_dists = sklearn.metrics.pairwise_distances(dists0, dists1, metric='l1')
            C_ref = dists1.size
            arg_ref = np.argmin(dists_ref_dists)
            min_ref_dist = np.min(dists_ref_dists)

            # if no good reference match, then skip for now (might come back later)
            if min_ref_dist > MIN_REF_DIST_THRESH_UPPER \
            or min_ref_dist < MIN_REF_DIST_THRESH_LOWER: # match too similar (e.g. duplicate image)
                matches_skipped.append({
                    'idx0': idx0,
                    'idx1': idx1,
                    'dist': min_ref_dist,
                })
                pairwise_constraints[idx0, idx1] = True
            else:
                ## pick best match
                arg_ref0 = arg_ref // C_ref
                arg_ref1 = arg_ref % C_ref
                idx_ref0 = np.where(idxs_ref0)[0][arg_ref0]
                idx_ref1 = np.where(idxs_ref1)[0][arg_ref1]

                ## save the pairs
                match = {
                    s0: idx0,
                    s1: idx1,
                    s0 + '_ref': idx_ref0,
                    s1 + '_ref': idx_ref1,
                    'dist': dists_match_constrained[arg0, arg1],
                    'dist_ref0': dists0[arg_ref0][0],
                    'dist_ref1': dists1[arg_ref1][0],
                }
                for k in match.keys():
                    if not k in matches:
                        matches[k] = []
                    matches[k].append(match[k])
                print('len(matches)', len(matches['dist']))

                ## remove them from further consideration
                idxs_to_remove = (df.id == id0) | (df.id == id1)
                subgroups[s0][idxs_to_remove] = False
                subgroups[s1][idxs_to_remove] = False
    #         print(matches, matches_skipped)
    #     print('num_matches', len(matches))
    return matches

def plot_subgroup_means(g0, g1, ks, ticklabels=True, args=None):
    '''
    args is used to ensure that yticks are put in same order
    '''
    if args is None:
        args = np.argsort(np.abs(g0[ks].mean() - g1[ks].mean()).values)
    for g, lab in zip([g0, g1], ['Perceived as female', 'Perceived as male']):
        means = g[ks].mean().values
        stds = 1.96 * g[ks].std().values / np.sqrt(g.shape[0])
        ys = np.arange(len(ks))
        plt.errorbar(means[args], ys, label=lab, xerr=stds[args],
                     linestyle='', marker='.', markersize=10)
        if ticklabels:
            plt.yticks(ys, [k.capitalize().replace('_', ' ') for k in ks[args]])
        else:
            plt.yticks(ys, ['' for k in ks[args]])
    plt.xlim((0, 1))
    plt.xlabel('Mean value in dataset')
    plt.grid()
    return args