import sys
from os.path import join as oj
import os
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('..')
from config import *
from copy import deepcopy
import pickle as pkl
import face_recognition


out_dir = oj(DIR_PROCESSED, 'processed_latents')
os.makedirs(out_dir, exist_ok=True)
# out_fname = oj(DIR_PROCESSED, 'processed_latents_300.pkl')
fname_faceid = oj(DIR_PROCESSED_MISC, f'facial_dists_after_projection.csv')
IM_NUMS = range(300)
regs = [0, 0.01, 0.1, 1, 10000]
ks = sorted(ATTRS_MEASURED)

if __name__ == '__main__':
    dists = {
        reg: [] for reg in regs
    }

    for r, im_num in tqdm(enumerate(IM_NUMS)):
        im_orig_fname = oj(DIR_IMS, f'{im_num + 1:05}.jpg')
        im_orig = face_recognition.load_image_file(im_orig_fname)
        im_orig_encodings = face_recognition.face_encodings(im_orig)
        if len(im_orig_encodings) == 0:
            continue
        else:
            im_orig_encoding = im_orig_encodings[0]

        for reg in regs:
            im_new_fname = oj(DIR_PROCESSED, f'generated_images_{reg}/{im_num + 1:05}.png')
            im_new = face_recognition.load_image_file(im_new_fname)
            try:
                im_new_encoding = face_recognition.face_encodings(im_new)[0]
                dists[reg].append(face_recognition.face_distance([im_orig_encoding], im_new_encoding)[0])
            except:
                dists[reg].append(np.nan)
            # results = face_recognition.compare_faces([im_orig_encoding], im_new_encoding)

    pd.DataFrame.from_dict(dists).to_csv(fname_faceid)
