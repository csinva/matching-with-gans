import tensorflow as tf
import sys
sys.path.append('../models/stylegan2')
import dnnlib
import dnnlib.tflib as tflib
from run_generator import generate_images
import pretrained_networks
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join as oj
import pickle as pkl

class Generator:
    def __init__(self):
        network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
        truncation_psi = 0.5
        
        # load the networks
        print('Loading networks from "%s"...' % network_pkl)
        _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi
        
        self.Gs = Gs
        self.Gs_kwargs = Gs_kwargs
    
    def gen(self, z):
        return self.Gs.run(z, None, **self.Gs_kwargs) # [minibatch, height, width, channel]