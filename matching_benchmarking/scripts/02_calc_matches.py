'''
we want to find matched pairs
the pairs should change only one attribute at a time (but we may vary 2 to make a transect)
"main" image is matched stringently
"reference" image is matched more loosely
'''

import sys

sys.path.append('..')
sys.path.append('../..')

import pandas as pd
import matching
from config import *
import data

NUM_MATCHES = 1200
FACIAL_REC_THRESH = 0.6 # 0.6 is classification threshold for dlib, None removes this constraint
MIN_REF_DIST_THRESH_UPPER = 1e6 # weeds out any crazy matches
MIN_REF_DIST_THRESH_LOWER = 1e-2 # weeds out any matches that are too close


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
dists_match = data.get_dists('gan') 

# implement the facial rec threshold constraint
# by making distances which violate the constraint extremely high
if FACIAL_REC_THRESH is not None:
    dists_match = dists_match + (data.get_dists('facial') > FACIAL_REC_THRESH) * 1e6 # constraint for missclassificaiton


# specify things to vary (these should all be binary columns)
# id willl automatically be different when we vary gender, race bc these things 
# are forced to be preserved in data.py
save_name = oj(DIR_PROCESSED, f'matches_{attrs_to_vary[0]}_num={NUM_MATCHES}_facerecthresh={FACIAL_REC_THRESH}.pkl')
matches = matching.get_matches(df, dists_match, dists_ref, attrs_to_vary,
                               NUM_MATCHES, MIN_REF_DIST_THRESH_UPPER, MIN_REF_DIST_THRESH_LOWER, save_name=save_name)
matches = pd.DataFrame.from_dict(matches).infer_objects()
matches.to_pickle(save_name)