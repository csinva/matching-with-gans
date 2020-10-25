from copy import deepcopy
from collections import OrderedDict
import numpy as np
import pandas as pd
import sklearn.metrics
from tqdm import tqdm


def get_lat(latents):
    #     lat = latents.mean(axis=1) # match in style space
    lat = latents.reshape(latents.shape[0], -1)  # match in extended style space
    return lat


def calc_dists_from_latents(lat):
    '''Get distances (points, points)
    '''
    dists = sklearn.metrics.pairwise_distances(lat, metric='l2')  # l2 dist
    dists[np.eye(dists.shape[0]).astype(bool)] = 1e3  # don't pick same point
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
                NUM_MATCHES, MIN_REF_DIST_THRESH_UPPER, MIN_REF_DIST_THRESH_LOWER,
                save_name, save_freq=50):
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
                
        # save the matches
        if match_num % save_freq == 0:
            df_save = pd.DataFrame.from_dict(matches).infer_objects()
            df_save.to_pickle(save_name.replace('num=', f'num={match_num}-'))
    return matches

def add_intersections(d, ks_init, ignore_key='gender'):
    ks_init = [k for k in ks_init if not k == ignore_key]
    ks = []
    for i, k1 in enumerate(ks_init):
        for j in range(i):
            k2 = ks_init[j]
            k_full = f'{k1.capitalize()} & {k2.capitalize()}'
            d[k_full] = np.array(d[k1] == 1) & np.array(d[k2] == 1)
            ks.append(k_full)
            
            k_full = f'{k1.capitalize()} & -{k2.capitalize()}'
            d[k_full] = np.array(d[k1] == 1) & np.array(d[k2] == 0)
            ks.append(k_full)
            
            k_full = f'-{k1.capitalize()} & {k2.capitalize()}'
            d[k_full] = np.array(d[k1] == 0) & np.array(d[k2] == 1)
            ks.append(k_full)
            
            k_full = f'-{k1.capitalize()} & -{k2.capitalize()}'
            d[k_full] = np.array(d[k1] == 0) & np.array(d[k2] == 0)
            ks.append(k_full)
            
    return d, ks



def calc_propensity_matches(groups, propensity, caliper = 0.05):
    ''' 
    Params
    ------
    groups
        Treatment assignments.  Must be 2 groups
    propensity
        Propensity scores for each observation. Propensity and groups should be in the same order (matching indices)
    caliper
        Maximum difference in matched propensity scores. For now, this is a caliper on the raw
        propensity; Austin reccommends using a caliper on the logit propensity.
    
    Returns
    -------
    matches1, matches2
        A series containing the individuals in the control group matched to the treatment group.
        Note that with caliper matching, not every treated individual may have a match.
    '''

    # Check inputs
    if any(propensity <=0) or any(propensity >=1):
        raise ValueError('Propensity scores must be between 0 and 1')
    elif not(0<caliper<1):
        raise ValueError('Caliper must be between 0 and 1')
    elif len(groups)!= len(propensity):
        raise ValueError('groups and propensity scores must be same dimension')
    elif len(groups.unique()) != 2:
        raise ValueError('wrong number of groups')
        
        
    # Code groups as 0 and 1
    groups = groups == groups.unique()[0]
    N = len(groups)
    N1 = groups.sum(); N2 = N-N1
    g1, g2 = propensity[groups == 1], (propensity[groups == 0])
    
    # Check if treatment groups got flipped - treatment (coded 1) should be the smaller
    if N1 > N2:
        N1, N2, g1, g2 = N2, N1, g2, g1 
        
    # Randomly permute the smaller group to get order for matching
    np.random.seed(42)
    idxs_shuffled = np.random.permutation(N1)
    matches1 = []
    matches2 = []
    
    # cycle through members of group 1
    for idx in tqdm(idxs_shuffled):
        
        # distance of each member of group2 to g1.iloc[idx]
        dist = abs(g1.iloc[idx] - g2).values
        
        # if match is below some thresh
        if dist.min() <= caliper:
            
            # pick the best match
            arg_min = np.argmin(dist)
    
            # append indexes of matches
            matches2.append(g2.index[arg_min])
            matches1.append(g1.index[idx])
            
            # don't consider this for future matches
            g2 = g2.drop(matches2[-1])
    return matches1, matches2

def matches_to_df(matches, df, k_group):
    # match indexes are in original space [0, 30000)
    # this is the same as the df.index [0, 30000)
    match_keys = OrderedDict({
        f'{k_group}_0_ref': 'dist_ref0',
        f'{k_group}_0': 'dist',
        f'{k_group}_1': 'dist',
        f'{k_group}_1_ref': 'dist_ref1'
    })
    ks_matched = [k for k in match_keys if not 'ref' in k]
    idxs_matched = matches[ks_matched].values
    df_matched = df.iloc[idxs_matched.flatten()]

    # add Race = Black
    df['Race=Black'] = 0
    df.loc[df['race4_pred'] == 'Black', 'Race=Black'] = 1
    df_matched['Race=Black'] = 0
    df_matched.loc[df_matched['race4_pred'] == 'Black', 'Race=Black'] = 1
    return df_matched, match_keys