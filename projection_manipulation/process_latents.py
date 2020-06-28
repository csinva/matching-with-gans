import sys
from os.path import join as oj

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('stylegan2')
import pretrained_networks
import projector
# sys.path.append('transects')
from transects import make_transects
from config import *

out_fname = 'processed/09_df_99.pkl'
IM_NUMS = np.arange(1, 99)
regs = [0, 0.01, 0.1, 1, 10000]
ks = sorted(ATTRS_MEASURED)

if __name__ == '__main__':
    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    _, _, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)

    coefs, intercepts = make_transects.get_directions(all_attrs=ALL_ATTRS)
    coefs = np.array(coefs).squeeze()
    intercepts = np.array(intercepts).flatten()

    # want df where each row is an (image, reg_param) set
    # columns represent different measure statistics
    r = {
        k: []
        for k in ['perceptual_loss', 'mean_abs_corr', 'im_num', 'reg_param'] +
                 [f'pred_{a}' for a in ALL_ATTRS]
        #         [f'lab_{a}' for a in ALL_ATTRS]
    }

    for im_num in tqdm(IM_NUMS):
        # load original image
        im_orig = mpimg.imread(oj(DIR_IMS, f'{im_num:05}.jpg'))
        im_orig = np.expand_dims(np.transpose(im_orig, (2, 0, 1)), 0)  # (1, 3, 1024, 1024)

        for reg in regs:

            r['reg_param'].append(reg)
            r['im_num'].append(im_num)

            # load latents
            folder = f'generated_images_{reg}'
            # im_gen_fname = oj(DIRS_GEN, folder, f'{im_num:05}.png')        
            latents = np.load(oj(DIR_PROCESSED, folder, f'{im_num:05}.npy'))
            latents = np.expand_dims(latents, 0)  # (1, 18, 512)

            # calculate losses
            r['perceptual_loss'].append(proj.get_vgg_loss(im_orig, latents))
            r['mean_abs_corr'].append(np.mean(np.abs(np.corrcoef(latents[0]))))

            # calculate predictions for each label
            latents_mean = np.mean(latents, axis=1)
            preds = (latents_mean @ coefs.transpose() + intercepts)
            #         preds = (preds > 0) * 1
            #         preds[preds == 0] = -1
            preds = preds.flatten()
            for i, a in enumerate(ALL_ATTRS):
                r[f'pred_{a}'].append(preds[i])
        #                 r[f'lab_{a}'].append(labs.iloc[im_num + 1][attr_map[a]])

        if im_num % 15 == 0:
            df = pd.DataFrame.from_dict(r)
            df.to_pickle(out_fname)
