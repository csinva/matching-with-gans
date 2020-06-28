import sys
from os.path import join as oj

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.append('..')
from config import *

sys.path.append(DIR_STYLEGAN)


out_fname = oj(DIR_PROCESSED, 'processed_latents_300.pkl')
IM_NUMS = range(300)
regs = [0, 0.01, 0.1, 1, 10000]
ks = sorted(ATTRS_MEASURED)

if __name__ == '__main__':
    import pretrained_networks
    import transects
    import projector
    
    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    _, _, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)

    coefs, intercepts = transects.get_directions(all_attrs=ALL_ATTRS)
    coefs = np.array(coefs).squeeze()
    intercepts = np.array(intercepts).flatten()

    # want df where each row is an (image, reg_param) set
    # columns represent different measure statistics
    r = {
        k: []
        for k in ['perceptual_loss', 'mean_abs_corr', 'mean_mse_dist', 'im_num', 'reg_param'] +
                 [f'pred_{a}' for a in ALL_ATTRS]
    }

    for im_num in tqdm(IM_NUMS):
        # load original image
        im_orig = mpimg.imread(oj(DIR_IMS, f'{im_num + 1:05}.jpg'))
        im_orig = np.expand_dims(np.transpose(im_orig, (2, 0, 1)), 0)  # (1, 3, 1024, 1024)

        for reg in regs:

            r['reg_param'].append(reg)
            r['im_num'].append(im_num)

            # load latents
            folder = f'generated_images_{reg}'
            # im_gen_fname = oj(DIRS_GEN, folder, f'{im_num:05}.png')        
            latents = np.load(oj(DIR_PROCESSED, folder, f'{im_num + 1:05}.npy'))
            latents = np.expand_dims(latents, 0)  # (1, 18, 512)
            latents_mean = latents.mean(axis=1)

            # calculate losses
            r['perceptual_loss'].append(proj.get_vgg_loss(im_orig, latents))
            r['mean_abs_corr'].append(np.mean(np.abs(np.corrcoef(latents[0]))))
            r['mean_mse_dist'].append(np.mean(np.square(latents - latents_mean)))

            # calculate predictions for each label
            latents_mean = np.mean(latents, axis=1)
            preds = (latents_mean @ coefs.transpose() + intercepts)
            preds = preds.flatten()
            for i, a in enumerate(ALL_ATTRS):
                r[f'pred_{a}'].append(preds[i])

        if im_num % 15 == 0:
            pd.DataFrame.from_dict(r).to_pickle(out_fname)
    pd.DataFrame.from_dict(r).to_pickle(out_fname)