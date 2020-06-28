import os
from os.path import join as oj

import face_recognition
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    DIR_ENCODINGS = '../data_processed/celeba-hq/encodings_dlib/'
    out_fname = 'processed/13_facial_dists_pairwise.npy'
    os.makedirs(DIR_ENCODINGS, exist_ok=True)

    # get fnames
    fnames = sorted([f for f in os.listdir(DIR_IMS) if '.jpg' in f])
    n = len(fnames)

    # calc encodings
    '''
    for i in tqdm(range(n)):
        fname_out = oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy'
        if not os.path.exists(fname_out):
            im = mpimg.imread(oj(DIR_IMS, fnames[i]))
            encoding = face_recognition.face_encodings(im, model='cnn')
            if len(encoding) > 0:
                encoding = encoding[0]
                np.save(open(fname_out, 'wb'), encoding)
            else:
                np.save(open(fname_out, 'wb'), np.zeros(128))
    '''

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
    print('loading encodings...')
    encodings = [np.load(open(oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy', 'rb'))
                 for i in range(n)]
    for i in tqdm(range(n)):
        if i in failures:
            continue
        encoding1 = encodings[i]
        for j in range(i):
            if j in failures:
                continue
            encoding2 = encodings[j]
            facial_dist = face_recognition.face_distance([encoding1], encoding2)[0]
            dists_facial[i, j] = facial_dist
            dists_facial[j, i] = facial_dist
        # if i % 1000 == 999:
        # np.save(open(out_fname, 'wb'), dists_facial)
        # pkl.dump({'facial_dists': dists_facial}, )
    dists_facial[np.eye(n).astype(bool)] = 1e3  # don't pick same point
    np.save(open(out_fname, 'wb'), dists_facial)
    # pkl.dump({'facial_dists': dists_facial, 'ids': fname_ids}, open(out_fname, 'wb'))
