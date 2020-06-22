import sys
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join as oj
import pandas as pd
import pickle as pkl
import sklearn.model_selection
import sklearn.metrics
import models
import util
import os
import config
from config import ATTR_TO_INDEX
import viz
import scipy.stats
from tqdm import tqdm
import figs
import matplotlib.image as mpimg
import seaborn as sns
import data
CELEB_IMS_DIR = '../data/celeba-hq/ims/'
CELEB_ANNO_DIR = '../data/celeba-hq/Anno/'
import face_recognition


if __name__ == '__main__':
    # load and merge all the data
    print('loading...')
    df = data.load_ids()
    labs, labs_full = data.load_labs()
    for k in labs.keys():
        df[k] = labs[k].values
    for k in labs_full.keys():
        df[k] = labs_full[k].values
    print('done loading!')

    out_fname = 'processed/12_facial_dists.pkl'
    d = df[df['count_with_this_id'] > 1]
    ids = sorted(d.id.unique())
    facial_dists = []
    ids_working = []
    print('num ids', len(ids))
    for i in tqdm(ids):
        try:
            ids = df[df.id == i][:2]
            ims = np.array([mpimg.imread(oj(CELEB_IMS_DIR, fname))
                            for fname in ids.fname_final.values])
            encoding1 = face_recognition.face_encodings(ims[0])[0]
            encoding2 = face_recognition.face_encodings(ims[1])[0]
            facial_dist = face_recognition.face_distance([encoding1], encoding2)[0]
            ids_working.append(i)
            facial_dists.append(facial_dist)
        except:
            pkl.dump({'facial_dists': facial_dists, 'ids': ids_working}, open(out_fname, 'wb')) # just randomly save sometimes
    pkl.dump({'facial_dists': facial_dists, 'ids': ids_working}, open(out_fname, 'wb'))