import pandas as pd
from os.path import join as oj
import os
import numpy as np
from tqdm import tqdm
import scipy.special
def load_all_labs(cached_file='processed/df.pkl',
                  CELEB_IMS_DIR = '../data/celeba-hq/ims/',
                  CELEB_ANNO_DIR = '../data/celeba-hq/Anno/',
                  celeba_labs_fname='../data/celeba-hq/Anno/list_attr_celeba.txt',
                  mapping_file='../data/celeba-hq/mapping.txt',
                  race_preds_file='../data_processed/celeba-hq/attr_preds/preds.pkl',
                  quality_scores_file='/home/ubuntu/face-disentanglement/data_processed/celeba-hq/quality_scores.pkl',
                  background_stats_file='processed/15_background_stats.pkl',
                  pose_file='/home/ubuntu/face-disentanglement/data_processed/celeba-hq/pose.pkl'
                 ):
    if os.path.exists(cached_file):
        print('loading cached labels')
        return pd.read_pickle(cached_file)
    
    print('loading labels...')
    df = load_ids(CELEB_IMS_DIR, CELEB_ANNO_DIR)
    labs, labs_full = load_labs(celeba_labs_fname, mapping_file)
    
    
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
        6329: 491 # same image, has different ids
    }
    for i in IDS_TO_MERGE:
        df.loc[df.id == i, 'id'] = IDS_TO_MERGE[i]
    
    
    # replace value for some attributes by the mode over all images with this id
    attrs = ['gender', 'race_pred', 'race4_pred']
    for attr in attrs:
        for i in tqdm(df.id.unique()):
            idxs = df.id == i
            mode = df[idxs][attr].mode().values[0] # get mode if there is disagreement
            df.loc[idxs, attr] = mode
            
    # cache the dataframe
    df.to_pickle(cached_file)
    return df

def load_labs(celeba_labs_fname, mapping_file, N_IMS=30000):
    '''Load labels for celeba-hq
    '''
    remap = pd.read_csv(mapping_file, delim_whitespace=True)
    labs_full = pd.read_csv(celeba_labs_fname, delim_whitespace=True, skiprows=1)
    
    labs_full = labs_full.loc[[remap.iloc[i]['orig_file'] for i in range(N_IMS)]] #for i in range(labs_full.shape[0])]
    labs_full = labs_full == 1
    # print(labs_full.head())
    labs = pd.DataFrame()
    # print(labs.keys())
    # print(labs_full.keys())

    # large is more male
    labs['gender'] = labs_full['Male']

    # larger is longer
    labs['hair-length'] = ~(labs_full['Bald'] | labs_full['Receding_Hairline']) # Bangs, Receding_Hairline

    # larger is more
    labs['facial-hair'] = (~(labs_full['No_Beard']) |  labs_full['Mustache'] | labs_full['Goatee']) # labs_full['Mustache'] # Goatee, Mustache, No_Beard, 5_o_Clock_Shadow

    # higher is more
    labs['makeup'] = labs_full['Heavy_Makeup'] # | labs_full['Wearing_Lipstick'] # Wearing_Lipstick

    # higher is darker
    labs['skin-color'] = labs_full['Pale_Skin']

    # older is more positive
    labs['age'] = ~labs_full['Young']

    # make into int
    labs = labs.astype(int)
    labs_full = labs_full.astype(int)
    
    return labs, labs_full

def load_ids(CELEB_IMS_DIR, CELEB_ANNO_DIR):
    '''Load IDs for celeba-hq
    '''
    ids_orig = pd.read_csv(oj(CELEB_ANNO_DIR, 'identity_CelebA.txt'), delim_whitespace=True, header=None)
    ids_orig = ids_orig.rename(columns={0: 'orig_file', 1: 'id'})
    remap = pd.read_csv('../data/celeba-hq/mapping.txt', delim_whitespace=True)
    
    
    # labels for celeb-a (not hq)
    # (vals, counts) = np.unique(ids_orig.id.values, return_counts=True)
    # plt.hist(counts)
    # plt.xlabel('number of images with same id\nin celeb-a (hq has less)')
    # plt.ylabel('count')
    # plt.show()
    
    
    fnames = sorted([f for f in os.listdir(oj(CELEB_IMS_DIR)) if '.jpg' in f])
    ids = remap.merge(ids_orig, on='orig_file', how='left')
    ids['fname_final'] = fnames
    (vals, counts) = np.unique(ids.id.values, return_counts=True)
    id_to_count = {v: c for (v, c) in zip(vals, counts)}
    ids['count_with_this_id'] = ids['id'].map(id_to_count)
    
    return ids