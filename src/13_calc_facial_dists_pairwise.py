import sys
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join as oj
import pandas as pd
import pickle as pkl
import models
import util
import os
import config
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
    DIR_ENCODINGS = '../data_processed/celeba-hq/encodings_dlib/'
    out_fname = 'processed/13_facial_dists_pairwise.pkl'
    os.makedirs(DIR_ENCODINGS, exist_ok=True)
    
    # get fnames
    fnames = sorted([f for f in os.listdir(DIR_ORIG) if '.jpg' in f])
    n = len(fnames)
    
    
    # calc encodings
    for i in tqdm(range(n)):
        fname_out = oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy'
        if not os.path.exists(fname_out):
            im = mpimg.imread(oj(DIR_ORIG, fnames[i]))
            encoding = face_recognition.face_encodings(im, model='cnn')
            if len(encoding) > 0:
                encoding = encoding[0]
                np.save(open(fname_out, 'wb'), encoding)
            else:
                np.save(open(fname_out, 'wb'), np.zeros(128))
                
    # calc failures
    FAILURES_FILE = oj(DIR_ENCODINGS, 'failures.npy')
    if not os.path.exists(FAILURES_FILE):
        failures = []
        for i in tqdm(range(n)):
            fname_out = oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy'
            encoding = np.load(open(fname_out, 'rb'))
            if not np.any(encoding):
                failures.append(i)
        np.save(open(FAILURES_FILE, 'wb'), np.array(failures))
    else:
        failures = np.load(open(FAILURES_FILE, 'rb'))
    
    # calc dists
    dists_facial = np.ones((n, n)) * 1e3
    for i in tqdm(range(n - 1)):
        if i in failures:
            continue
        fname_out = oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy'
        encoding1 = np.load(open(fname_out, 'rb'))
        for j in tqdm(range(i + 1, n)):
            if j in failures:
                continue
            fname_out = oj(DIR_ENCODINGS, fnames[j][:-4]) + '.npy'
            encoding2 = np.load(open(fname_out, 'rb'))
            facial_dist = face_recognition.face_distance([encoding1], encoding2)[0]
            dists_facial[i, j] = facial_dist
            dists_facial[j, i] = facial_dist
        if i % 1000 == 999:
            pkl.dump({'facial_dists': dists_facial, 'ids': fname_ids}, open(out_fname, 'wb'))
    dists_facial[np.eye(n).astype(bool)] = 1e3 # don't pick same point
    pkl.dump({'facial_dists': dists_facial, 'ids': fname_ids}, open(out_fname, 'wb'))