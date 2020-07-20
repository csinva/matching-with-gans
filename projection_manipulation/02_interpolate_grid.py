'''This script makes a grid which interpolates halfway between all pairs
'''

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
from ganwrapper import Generator
from tqdm import tqdm

import util
from config import *
import config


if __name__ == '__main__':
    ALIGNED_IMAGES_DIR = sys.argv[1]
    INTERPOLATED_IMAGES_DIR = sys.argv[2]
    N_IMS = 11
    names = None
    names = ['chandan', 'andy', 'varun']
#     names = ['chandan', 'dad', 'amma']

    G = Generator(image_size=1024)
    os.makedirs(INTERPOLATED_IMAGES_DIR, exist_ok=True)

    fnames = sorted([f for f in os.listdir(ALIGNED_IMAGES_DIR)
                     if '.npy' in f])
    
    
    
    
    # filter the names
    if names is not None:
        print('filtering names!')
        fnames_temp = set()
        for fname in fnames:
            for name in names:
                if name in fname:
                    fnames_temp.add(fname)
        fnames = sorted(list(fnames_temp))

    latents = np.array([np.load(oj(ALIGNED_IMAGES_DIR, f)) for f in fnames])
    ims_list = []
    n = len(fnames)
    
    # top row
    ims_list.append(np.zeros((5, 5)))
    for col in tqdm(range(n)):
        ims_list.append(G.generateImageFromStyleFull(latents[col: col + 1]))
        
    # other rows
    for row in tqdm(range(n)):
        # add back orignal im on y axis
        ims_list.append(ims_list[row + 1]) 
        for col in range(n):
            ims_list.append(G.generateImageFromStyleFull(0.5 * latents[row: row + 1] + 0.5 * latents[col: col + 1]))
            
    
    ims = np.array(ims_list).reshape((n + 1, n + 1))
    
    util.plot_grid(ims)
    fname_out = oj(INTERPOLATED_IMAGES_DIR, fnames[0][:-4] + '-' + fnames[-1][:-4] + '_grid' + '.png')
    plt.savefig(fname_out, dpi=150)
    print('grid saved!')