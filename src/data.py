import pandas as pd
from os.path import join as oj
import os
import numpy as np
from tqdm import tqdm

def load_all_labs(preds_file='../data_processed/celeba-hq/attr_preds/preds.pkl', cached_file='processed/df.pkl'):
    if os.path.exists(cached_file):
        print('loading cached labels')
        return pd.read_pickle(cached_file)
    
    print('loading labels...')
    df = load_ids()
    labs, labs_full = load_labs()
    pred_labs = pd.read_pickle(preds_file)
    for k in labs.keys():
        df[k] = labs[k].values
    for k in labs_full.keys():
        df[k] = labs_full[k].values
    for k in pred_labs.keys():
        df[k + '_pred'] = pred_labs[k].values
    df['fname_id'] = df['fname_final'].str.slice(stop=-4)
    
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

def load_labs(N_IMS=30000):
    '''Load labels for celeba-hq
    '''
    celeba_labs_fname='../data/celeba-hq/Anno/list_attr_celeba.txt'
    remap = pd.read_csv('../data/celeba-hq/mapping.txt', delim_whitespace=True)
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
    labs['facial-hair'] = ~(~(labs_full['No_Beard']) |  labs_full['Mustache'] | labs_full['Goatee']) # labs_full['Mustache'] # Goatee, Mustache, No_Beard, 5_o_Clock_Shadow

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

def load_ids(CELEB_IMS_DIR = '../data/celeba-hq/ims/',
             CELEB_ANNO_DIR = '../data/celeba-hq/Anno/'):
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