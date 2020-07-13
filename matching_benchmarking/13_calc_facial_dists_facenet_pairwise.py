import os
from os.path import join as oj
import sys
sys.path.append('..')
sys.path.append('../lib/facenet/src')
from config import *
import facenet

import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg
import skimage.transform

if __name__ == '__main__':
    DIR_ENCODINGS = oj(DIR_PROCESSED, 'encodings_facenet_casia') # VGGFace2
    out_fname = oj(DIR_PROCESSED, 'dists_pairwise_facial_facenet_casia.npy') # CASIA-WebFace
    model_path = '/home/ubuntu/face-disentanglement/lib/facenet/20180408-102900' # CASIA-WebFace    
    
#     DIR_ENCODINGS = oj(DIR_CELEBA, 'encodings_facenet') # VGGFace2
#     out_fname = oj(DIR_PROCESSED, 'dists_pairwise_facial_facenet_vgg2.npy') # VGGFace2
#     model_path = '/home/ubuntu/face-disentanglement/lib/facenet/20180402-114759' # VGGFace2
    os.makedirs(DIR_ENCODINGS, exist_ok=True)
    
    # get fnames
    fnames = sorted([f for f in os.listdir(DIR_IMS) if '.jpg' in f])
    n = len(fnames)
    
    
    import tensorflow as tf
    with tf.Graph().as_default():
        with tf.Session() as sess:
            
            # load the model
            facenet.load_model(model_path)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # calc encodings
            for i in tqdm(range(n)):
                fname_out = oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy'
                if not os.path.exists(fname_out):
                    im = mpimg.imread(oj(DIR_IMS, fnames[i]))
                    im = skimage.transform.resize(im, (160, 160, 3))
                    # Run forward pass to calculate embeddings
                    feed_dict = {
                        images_placeholder: [im],
                        phase_train_placeholder: False
                    }
                    encoding = sess.run(embeddings, feed_dict=feed_dict)
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
            facial_dist = np.sqrt(np.sum(np.square(np.subtract(encoding1, encoding2))))
            dists_facial[i, j] = facial_dist
            dists_facial[j, i] = facial_dist
        # if i % 1000 == 999:
        # np.save(open(out_fname, 'wb'), dists_facial)
        # pkl.dump({'facial_dists': dists_facial}, )
    dists_facial[np.eye(n).astype(bool)] = 1e3  # don't pick same point
    np.save(open(out_fname, 'wb'), dists_facial)
    # pkl.dump({'facial_dists': dists_facial, 'ids': fname_ids}, open(out_fname, 'wb'))