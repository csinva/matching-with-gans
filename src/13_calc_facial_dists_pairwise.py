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
import face_recognition


if __name__ == '__main__':
    DIR_ORIG = '../data/celeba-hq/ims/'
    out_fname = 'processed/13_facial_dists_pairwise.pkl'
    
    
    DIRS_GEN = '../data_processed/celeba-hq/'
    reg = 0.1
    DIR_GEN = oj(DIRS_GEN, f'generated_images_{reg}')

    # get fnames
    fname_nps = [f for f in sorted(os.listdir(DIR_GEN)) if 'npy' in f]
    fname_ids = np.array([f[:-4] for f in fname_nps])
    n = fname_ids.size
    dists_facial = np.zeros((n, n))
    for i in tqdm(range(n)):
        im1 = mpimg.imread(oj(DIR_ORIG, f'{fname_ids[i]}.jpg'))
        encoding1 = face_recognition.face_encodings(im1)[0]
        for j in range(i):
            im2 = mpimg.imread(oj(DIR_ORIG, f'{fname_ids[j]}.jpg'))
            encoding2 = face_recognition.face_encodings(im2)[0]
            facial_dist = face_recognition.face_distance([encoding1], encoding2)[0]
            dists_facial[i, j] = facial_dist
            dists_facial[j, i] = facial_dist
        if i % 1000 == 999:
            pkl.dump({'facial_dists': dists_facial, 'ids': fname_ids}, open(out_fname, 'wb'))
    dists_facial[np.eye(n).astype(bool)] = 1e3 # don't pick same point
    pkl.dump({'facial_dists': dists_facial, 'ids': fname_ids}, open(out_fname, 'wb'))