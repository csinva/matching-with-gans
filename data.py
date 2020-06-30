import os

import numpy as np
import pandas as pd
import scipy.special
from tqdm import tqdm

from config import *


def get_dists(dist_type: str='facial'):
    '''
    dist_type
    '''
    dists_fnames = {
        'facial': oj(DIR_PROCESSED, 'dists_pairwise_facial.npy'),
        'vgg': oj(DIR_PROCESSED, 'dists_pairwise_vgg.npy'),
    }
    return np.load(open(dists_fnames[dist_type], 'rb'))
    
    
def load_all_labs(cached_file=oj(DIR_PROCESSED, 'df.pkl'),
                  dir_ims=oj(DIR_CELEBA, 'ims'),
                  celeba_id_fname=oj(DIR_CELEBA, 'Anno', 'identity_CelebA.txt'),
                  celeba_attr_fname=oj(DIR_CELEBA, 'Anno', 'list_attr_celeba.txt'),
                  mapping_file=oj(DIR_CELEBA, 'mapping.txt'),
                  race_preds_file=oj(DIR_PROCESSED, 'race.pkl'),
                  quality_scores_file=oj(DIR_PROCESSED, 'quality_scores.pkl'),
                  background_stats_file=oj(DIR_PROCESSED, 'background_stats.pkl'),
                  pose_file=oj(DIR_PROCESSED, 'pose.pkl')):

    if os.path.exists(cached_file):
        print('loading cached labels')
        return pd.read_pickle(cached_file)

    print('loading labels...')
    df = load_ids(dir_ims, celeba_id_fname, mapping_file)
    labs, labs_full = load_labs(celeba_attr_fname, mapping_file)

    # load in auxilary properties
    race_pred_labs = pd.read_pickle(race_preds_file)
    quality = pd.read_pickle(quality_scores_file)
    background = pd.read_pickle(background_stats_file)
    pose = pd.read_pickle(pose_file)

    for k in labs.keys():
        df[k] = labs[k].values
    for k in labs_full.keys():
        df[k] = labs_full[k].values
    for k in race_pred_labs.keys():
        df[k + '_pred'] = race_pred_labs[k].values
    for k in background.keys():
        df['background_' + k] = background[k].values

    # process and add head pose angles
    angles = np.array([ang for ang in range(66)])
    for k_new, k_orig in zip(['yaw', 'pitch', 'roll'], ['yaws', 'pitches', 'rolls']):
        vals = []
        for i in tqdm(range(df.shape[0])):
            arr = pose.iloc[i][k_orig].flatten()
            preds = scipy.special.softmax(arr)
            vals.append(np.sum(preds * angles) * 3 - 99)
        df[k_new] = vals

    df['quality'] = [s[0, 0] for s in quality['scores']]
    df['fname_id'] = df['fname_final'].str.slice(stop=-4)
    for i, race in enumerate(['White', 'Black', 'Asian', 'Indian']):
        df[race + '_prob'] = [x[i] for x in df['race_scores_fair_4_pred'].values]

    # clean up some labels
    # remove id errors (can eventually move this into data.py)
    # fix wrongly split ids
    IDS_TO_MERGE = {
        6329: 491  # same image, has different ids
    }
    for i in IDS_TO_MERGE:
        df.loc[df.id == i, 'id'] = IDS_TO_MERGE[i]

    # replace value for some attributes by the mode over all images with this id
    attrs = ['gender', 'race_pred', 'race4_pred']
    for attr in attrs:
        for i in tqdm(df.id.unique()):
            idxs = df.id == i
            mode = df[idxs][attr].mode().values[0]  # get mode if there is disagreement
            df.loc[idxs, attr] = mode

    # remove some unnecesary keys
    df.columns = df.columns.str.lower()
    df = df[[k for k in df.keys()
             if not 'md5' in k and not 'scores' in k
             and not 'idx' in k and not 'orig_file' in k
             and not 'img_names_pred' in k
             and not 'face_name_align_pred' in k]]
    df.keys()
            
            
    # cache the dataframe
    df.to_pickle(cached_file)
    df.to_csv(cached_file[:-4] + '.csv')
    return df


def load_labs(celeba_attr_fname, mapping_file, N_IMS=30000):
    '''Load labels for celeba-hq
    '''
    remap = pd.read_csv(mapping_file, delim_whitespace=True)
    labs_full = pd.read_csv(celeba_attr_fname, delim_whitespace=True, skiprows=1)

    labs_full = labs_full.loc[[remap.iloc[i]['orig_file'] for i in range(N_IMS)]]  # for i in range(labs_full.shape[0])]
    labs_full = labs_full == 1
    # print(labs_full.head())
    labs = pd.DataFrame()
    # print(labs.keys())
    # print(labs_full.keys())

    # large is more male
    labs['gender'] = labs_full['Male']

    # larger is longer
    labs['hair-length'] = ~(labs_full['Bald'] | labs_full['Receding_Hairline'])  # Bangs, Receding_Hairline

    # larger is more
    labs['facial-hair'] = ~(labs_full['No_Beard']) | labs_full['Mustache'] | labs_full['Goatee'] 
    # labs_full['Mustache'] # Goatee, Mustache, No_Beard, 5_o_Clock_Shadow

    # higher is more
    labs['makeup'] = labs_full['Heavy_Makeup']  # | labs_full['Wearing_Lipstick'] # Wearing_Lipstick

    # higher is darker
    labs['skin-color'] = labs_full['Pale_Skin']

    # older is more positive
    labs['age'] = ~labs_full['Young']

    # make into int
    labs = labs.astype(int)
    labs_full = labs_full.astype(int)

    return labs, labs_full


def load_ids(dir_ims, celeba_id_fname, mapping_file):
    '''Load IDs for celeba-hq
    '''
    ids_orig = pd.read_csv(celeba_id_fname, delim_whitespace=True, header=None)
    ids_orig = ids_orig.rename(columns={0: 'orig_file', 1: 'id'})
    remap = pd.read_csv(mapping_file, delim_whitespace=True)

    # labels for celeb-a (not hq)
    # (vals, counts) = np.unique(ids_orig.id.values, return_counts=True)
    # plt.hist(counts)
    # plt.xlabel('number of images with same id\nin celeb-a (hq has less)')
    # plt.ylabel('count')
    # plt.show()

    fnames = sorted([f for f in os.listdir(oj(dir_ims)) if '.jpg' in f])
    ids = remap.merge(ids_orig, on='orig_file', how='left')
    ids['fname_final'] = fnames
    (vals, counts) = np.unique(ids.id.values, return_counts=True)
    id_to_count = {v: c for (v, c) in zip(vals, counts)}
    ids['count_with_this_id'] = ids['id'].map(id_to_count)

    return ids
