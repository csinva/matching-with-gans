'''This script takes projected latents and changes them along
a few predfined axes (e.g. skin color, age, hair-color, gender)
'''

import sys

import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join as oj

dir_curr = os.path.dirname(os.path.realpath(__file__))
sys.path.append(oj(dir_curr, '../..'))
sys.path.append(oj(dir_curr, '..'))
import config
import transects
from ganwrapper import Generator
import util

if __name__ == '__main__':
    ALIGNED_IMAGES_DIR = sys.argv[1]
    MANIPULATED_IMAGES_DIR = sys.argv[2]
    if len(sys.argv) > 3:
        ATTRS = sys.argv[3]
    else:
        ATTRS = 'ACHG'  # HAGCBMSEW # CHG
    N_IMS = 11

    G = Generator(image_size=1024)
    os.makedirs(MANIPULATED_IMAGES_DIR, exist_ok=True)
    for fname in sorted([f for f in os.listdir(ALIGNED_IMAGES_DIR)
                         if '.npy' in f]):
        fname_out = oj(MANIPULATED_IMAGES_DIR, fname[:-4] + '.png')

        # skip if it already exists
        # if os.path.exists(fname_out):
        #     continue

        latents = np.array([np.load(oj(ALIGNED_IMAGES_DIR, fname))])

        kwargs = {
            # change these
            'save_dir': 'results/tnew',
            'latents': latents,

            # probably not these
            'G': G,
            'model_dir': config.DIR_LINEAR_DIRECTIONS,
            'orth': True,
            'randomize_seeds': False,
            'return_ims': True
        }

        # make 1D transects
        LIMS = {
            'C': [-1.5, 1.5],
            'H': [-0.5, 0.5],  # hair-length really shouldn't go beyond 0
            'G': [-1.75, 1.75],

            # these are not calibrated
            'A': [-2, 2],
            'B': [-2, 2],
            'M': [-2, 2],
            'S': [-2, 2],
            'E': [-2, 2],
            'W': [-2, 2],
        }

        transects_1d = {}
        for attr in ATTRS:
            ims, attr_df = transects.make_transects(
                attr=attr,
                N_IMS_LIST=[N_IMS],
                LIMS_LIST=LIMS[attr],
                force_project_to_boundary=False,
                **kwargs
            )
            transects_1d[attr] = ims

        # custom latent viz
        ims = np.array([transects_1d[a] for a in ATTRS])
        ims = ims.reshape((len(ATTRS), N_IMS, *ims.shape[2:]))
        util.plot_grid(ims, ylabs=[config.ATTR_LABELS[a].capitalize() for a in ATTRS], suptitle='Original')
        plt.savefig(fname_out, dpi=100)
        plt.close()
