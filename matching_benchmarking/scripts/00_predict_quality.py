import sys

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm import tqdm

from config import *

sys.path.append(oj(DIR_LIB, 'FaceQnet/src'))

fname_out = oj(DIR_CELEBA, 'quality_scores.pkl')

fnames = sorted([fname for fname in os.listdir(DIR_IMS)
                 if 'jpg' in fname])
print('fnames', len(fnames), fnames[:5], '...', fnames[-5:])
model = load_model('FaceQnet_v1.h5')
scores = []
for fname in tqdm(fnames):
    fname_full = os.path.join(DIR_IMS, fname)
    X_test = [cv2.resize(cv2.imread(fname_full, cv2.IMREAD_COLOR), (224, 224))]
    test_data = np.array(X_test, copy=False, dtype=np.float32)
    scores.append(model.predict(test_data, batch_size=1, verbose=0))

pd.DataFrame.from_dict({
    'fnames': fnames,
    'scores': scores
}).to_pickle(fname_out)
