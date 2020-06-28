import os
import sys

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm

sys.path.append('../lib/FaceQnet/src')

image_path = "/home/ubuntu/face-disentanglement/data/celeba-hq/ims/"
fname_out = '/home/ubuntu/face-disentanglement/data_processed/celeba-hq/quality_scores.pkl'

fnames = sorted([fname for fname in os.listdir(image_path)
                 if 'jpg' in fname])
print('fnames', len(fnames), fnames[:5], '...', fnames[-5:])
model = load_model('FaceQnet_v1.h5')
scores = []
for fname in tqdm(fnames):
    fname_full = os.path.join(image_path, fname)
    X_test = [cv2.resize(cv2.imread(fname_full, cv2.IMREAD_COLOR), (224, 224))]
    test_data = np.array(X_test, copy=False, dtype=np.float32)
    scores.append(model.predict(test_data, batch_size=1, verbose=0))

pd.DataFrame.from_dict({
    'fnames': fnames,
    'scores': scores
}).to_pickle(fname_out)
