# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import os
import cv2
import pandas as pd
from tqdm import tqdm

batch_size = 1 #Numero de muestras para cada batch (grupo de entrada)
image_path = "/home/ubuntu/face-disentanglement/data/celeba-hq/ims/"


fnames = sorted([fname for fname in os.listdir(image_path)
               if 'jpg' in fname])
print('fnames', len(fnames), fnames[:5], '...', fnames[-5:])
model = load_model('FaceQnet_v1.h5')
scores = []
for fname in tqdm(fnames):
    fname_full = os.path.join(image_path, fname)
    X_test = [cv2.resize(cv2.imread(fname_full, cv2.IMREAD_COLOR), (224, 224))]
    test_data = np.array(X_test, copy=False, dtype=np.float32)
    scores.append(model.predict(test_data, batch_size=batch_size, verbose=0))

pd.DataFrame.from_dict({
    'fnames': fnames,
    'scores': scores
}).to_pickle('./quality_scores.pkl')