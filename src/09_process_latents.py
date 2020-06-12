import sys
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join as oj
import pandas as pd
import pickle as pkl
import sklearn.model_selection
import models
import util
import os
import config
import viz
import scipy.stats
from tqdm import tqdm
import figs
import matplotlib.image as mpimg

sys.path.append('../models/stylegan2encoder')
import pretrained_networks
import projector
sys.path.append('transects')
from transects import make_transects, ganwrapper


IM_NUMS = np.arange(1, 99)
regs = [0, 0.1, 1, 10000]
attr_map = {
        'A': 'age',
        'B': 'facial-hair',
        'C': 'skin-color',
        'G': 'gender',
        'H': 'hair-length',
        'M': 'makeup',
    }
ks = sorted(attr_map.keys())

if __name__ == '__main__':
    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    _, _, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.set_network(Gs)

    DIR_ORIG = '../data/celeba-hq/ims/'
    DIRS_GEN = '../data_processed/celeba-hq/'

    all_attrs = ''.join(ks)
    coefs, intercepts = make_transects.get_directions(all_attrs=all_attrs)
    coefs = np.array(coefs).squeeze()
    intercepts = np.array(intercepts).flatten()


    def load_labs(celeba_labs_fname='../data/celeba-hq/Anno/list_attr_celeba.txt'):
        labs_full = pd.read_csv(celeba_labs_fname, delim_whitespace=True, skiprows=1)
        labs = pd.DataFrame()
        # print(labs.keys())
        # print(labs_full.keys())

        # large is more male
        labs['gender'] = labs_full['Male']

        # larger is longer
        labs['hair-length'] = -1 * labs_full['Bald'] # Bangs, Receding_Hairline

        # larger is more
        labs['facial-hair'] = labs_full['Mustache'] # Goatee, Mustache, No_Beard, 5_o_Clock_Shadow

        # higher is more
        labs['makeup'] = labs_full['Heavy_Makeup'] # Wearing_Lipstick

        # higher is darker
        labs['skin-color'] = labs_full['Pale_Skin'] * -1

        # older is more positive
        labs['age'] = labs_full['Young'] * -1
        return labs

    labs = load_labs()

    # want df where each row is an (image, reg_param) set
    # columns represent different measure statistics
    r = {
        k: []
        for k in ['perceptual_loss', 'mean_abs_corr', 'im_num', 'reg_param'] +
        [f'pred_{a}' for a in all_attrs] +
        [f'lab_{a}' for a in all_attrs]
    }

    for im_num in tqdm(IM_NUMS):
        # load original image
        im_orig = mpimg.imread(oj(DIR_ORIG, f'{im_num:05}.jpg'))
        im_orig = np.expand_dims(np.transpose(im_orig, (2, 0, 1)), 0) # (1, 3, 1024, 1024)

        for reg in regs:

            r['reg_param'].append(reg)
            r['im_num'].append(im_num)

            # load latents
            folder = f'generated_images_{reg}'
            # im_gen_fname = oj(DIRS_GEN, folder, f'{im_num:05}.png')        
            latents = np.load(oj(DIRS_GEN, folder, f'{im_num:05}.npy'))
            latents = np.expand_dims(latents, 0) # (1, 18, 512)        

            # calculate losses
            r['perceptual_loss'].append(proj.get_vgg_loss(im_orig, latents))
            r['mean_abs_corr'].append(np.mean(np.abs(np.corrcoef(latents[0]))))

            # calculate predictions for each label
            latents_mean = np.mean(latents, axis=1)
            preds = (latents_mean @ coefs.transpose() + intercepts)
    #         preds = (preds > 0) * 1
    #         preds[preds == 0] = -1
            preds = preds.flatten()
            for i, a in enumerate(all_attrs):
                r[f'pred_{a}'].append(preds[i])
                r[f'lab_{a}'].append(labs.iloc[im_num + 1][attr_map[a]])
    df = pd.DataFrame.from_dict(r)
    df.to_pickle('processed/09_df_100.pkl')