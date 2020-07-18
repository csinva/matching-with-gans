'''Note: this script was used to generate the results for face benchmarking the projections
A faster way to get similar results is to use the pre-calculated facial_dists
'''

import pickle as pkl

import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm

import data
from config import *

import face_recognition

if __name__ == '__main__':
    # load and merge all the data
    df = data.load_all_labs()

    out_fname = 'processed/12_facial_dists.pkl'
    d = df[df['count_with_this_id'] > 1]
    ids = sorted(d.id.unique())
    facial_dists = []
    ids_working = []
    print('num ids', len(ids))
    for i in tqdm(ids):
        try:
            ids = df[df.id == i][:2]
            ims = np.array([mpimg.imread(oj(DIR_IMS, fname))
                            for fname in ids.fname_final.values])
            encoding1 = face_recognition.face_encodings(ims[0])[0]
            encoding2 = face_recognition.face_encodings(ims[1])[0]
            facial_dist = face_recognition.face_distance([encoding1], encoding2)[0]
            ids_working.append(i)
            facial_dists.append(facial_dist)
        except:
            pkl.dump({'facial_dists': facial_dists, 'ids': ids_working},
                     open(out_fname, 'wb'))  # just randomly save sometimes
    pkl.dump({'facial_dists': facial_dists, 'ids': ids_working}, open(out_fname, 'wb'))
