'''
we want to find matched pairs
the pairs should change only one attribute at a time (but we may vary 2 to make a transect)
"main" image is matched stringently
"reference" image is matched more loosely
'''

import os
from copy import deepcopy
import sys
from os.path import join as oj
sys.path.append('..')
sys.path.append('../..')

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics
import matching
from tqdm import tqdm
from config import *
import data
import util


NUM_MATCHES = 200
MIN_REF_DIST_THRESH_UPPER = 1e6 # 1 will weed out any crazy matches
MIN_REF_DIST_THRESH_LOWER = 1e-2 # 1 will weed out any matches that are too close


# df contains filenames, ids, and attributes
df = data.load_all_labs()
n = df.shape[0]
df = df.set_index('fname_id')
df = df[['id', 'fname_final', 'age', 'gender', 'race_pred',
         'race4_pred', 'eyeglasses', 'black_or_white']] # only keep few keys to speed things up

attrs_to_vary = ['gender'] # gender, black_or_white

# specify dists for matching
print('loading / specifying dists...')
dists_match_name = 'gan_constrained'
dists_ref = data.get_dists('vgg')
dists_match = data.get_dists('gan') + (data.get_dists('facial') > 0.6) * 1e6 # constraint for missclassificaiton


# specify things to vary (these should all be binary columns)
# id willl automatically be different when we vary gender, race bc these things 
# are forced to be preserved in data.py
matches = matching.get_matches(df, dists_match, dists_ref, attrs_to_vary,
                               NUM_MATCHES, MIN_REF_DIST_THRESH_UPPER, MIN_REF_DIST_THRESH_LOWER)

matches = pd.DataFrame.from_dict(matches).infer_objects()
matches.to_pickle(oj(DIR_PROCESSED, f'matches_{attrs_to_vary[0]}_{matches.shape[0]}.pkl'))