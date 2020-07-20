'''This script interpolates between latents for pairs of images
in a given directory
'''

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('..')
import transects
from ganwrapper import Generator
from tqdm import tqdm

import util
from config import *
import config


if __name__ == '__main__':
    ALIGNED_IMAGES_DIR = sys.argv[1]
    INTERPOLATED_IMAGES_DIR = sys.argv[2]
    N_IMS = 11

    G = Generator(image_size=1024)
    os.makedirs(INTERPOLATED_IMAGES_DIR, exist_ok=True)

    fnames = [f for f in os.listdir(ALIGNED_IMAGES_DIR)
              if '.npy' in f]
    for i in range(len(fnames)):
        for j in range(i):
            fname_0 = fnames[i]
            fname_1 = fnames[j]
            fname_out = oj(INTERPOLATED_IMAGES_DIR, fname_0[:-4] + '_' + fname_1[:-4] + '.png')
            print(i, j, fnames[i], fnames[j])
            
            # skip if it already exists
            if os.path.exists(fname_out):
                continue
            
            # linearly interpolate between latents
            alphas = np.linspace(0, 1, N_IMS)
            alphas = alphas / np.max(alphas)
            latents0 = np.array([np.load(oj(ALIGNED_IMAGES_DIR, fname_0))])
            latents1 = np.array([np.load(oj(ALIGNED_IMAGES_DIR, fname_1))])
            latents = np.vstack([alpha * latents0 + (1 - alpha) * latents1
                                 for alpha in alphas])
            ims = G.generateImageFromStyleFull(latents)
            util.plot_row(ims)
            plt.savefig(fname_out, dpi=150)
            plt.close()