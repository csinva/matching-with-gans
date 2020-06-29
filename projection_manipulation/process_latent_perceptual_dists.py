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
sys.path.append(DIR_STYLEGAN)
import pickle as pkl


out_dir = oj(DIR_PROCESSED, 'processed_latents')
os.makedirs(out_dir, exist_ok=True)
# out_fname = oj(DIR_PROCESSED, 'processed_latents_300.pkl')
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
    for im_num in tqdm(IM_NUMS):
        
        
        # load original image
        im_orig = mpimg.imread(oj(DIR_IMS, f'{im_num + 1:05}.jpg'))
        im_orig = np.expand_dims(np.transpose(im_orig, (2, 0, 1)), 0)  # (1, 3, 1024, 1024)
        
        
        for reg in regs:
            out_fname = oj(out_dir, f'{im_num:05d}_{reg:0.3f}.pkl')
            if os.path.exists(out_fname):
                print('skipping', out_fname)
                continue
            r = {}
            r['reg_param'] = reg
            r['im_num'] = im_num

            # load latents
            folder = f'generated_images_{reg}'
            # im_gen_fname = oj(DIRS_GEN, folder, f'{im_num:05}.png')        
            latents = np.load(oj(DIR_PROCESSED, folder, f'{im_num + 1:05}.npy'))
            latents = np.expand_dims(latents, 0)  # (1, 18, 512)
            latents_mean = np.mean(latents, axis=1)

            # calculate losses
            r['perceptual_loss'] = proj.get_vgg_loss(im_orig, latents)
            r['mean_abs_corr'] = np.mean(np.abs(np.corrcoef(latents[0])))
            r['mean_mse_dist'] = np.mean(np.square(latents - latents_mean))

            # calculate predictions for each label
            preds = (latents_mean @ coefs.transpose() + intercepts)
            preds = preds.flatten()
            for i, a in enumerate(ALL_ATTRS):
                r[f'pred_{a}'] = preds[i]
        
            pkl.dump(r, open(out_fname, 'wb'))