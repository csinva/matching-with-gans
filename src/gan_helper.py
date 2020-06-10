import tensorflow as tf
import sys
sys.path.append('../models/stylegan2')
import dnnlib
import dnnlib.tflib as tflib
from run_generator import generate_images
import pretrained_networks
from training import misc
import projector
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join as oj
import pickle as pkl
import util
from copy import deepcopy

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
        
        # store generator
        self.Gs = Gs
        self.Gs_kwargs = Gs_kwargs
        
        # object for projecting real images
        self.proj = projector.Projector()
        self.proj.set_network(Gs)
        
    def z_to_style(self, z):
        '''Maps z to style vector
        Params
        ------
        z: np.ndarray
            (batch_size, 512)
        
        Returns
        -------
        w: np.ndarray
            (batch_size, 18, 512)
        '''
        return self.Gs.components.mapping.run(z, None, randomize_noise=False) # [minibatch, layer, component]
    
    def gen(self, z):
        '''Generate image from shared latent
        Params
        ------
        z: np.ndarray
            (batch_size, 512)
        '''        
        return self.Gs.run(z, None, **self.Gs_kwargs) # [minibatch, height, width, channel]
    
    def gen_full(self, z_full):
        '''Generate image from full latent
        Params
        ------
        z: np.ndarray
            (batch_size, 18, 512)
        '''
#         im = tflib.run(self.Gs.components.synthesis.get_output_for(z_full))
        im = self.Gs.components.synthesis.run(z_full, **self.Gs_kwargs)
        return im #get_transformed_im(im)
        
    def gen_full_old(self, z_full):
        '''Generate image from full latent
        Params
        ------
        z: np.ndarray
            (batch_size, 18, 512)
        '''
        im = tflib.run(self.Gs.components.synthesis.get_output_for(z_full))
        return get_transformed_im(im)        
        
    
    def project(self, im: np.ndarray, image_prefix='im0', num_steps=50, num_snapshots=3, lr=0.1):
        '''Projects an image into the latent space
        
        Params
        ------
        im
            (1, H, W, channels)
            
        Returns
        -------
        latents
            (batch_dim, 18, 512)
        image: np.ndarray
            (batch_dim, 1024, 1024, 3)
            
        also saves a bunch of things to the projections folder...
        '''            
            
        # adjust image range to [-1, 1]
        if np.max(im) > 1:
            im = (im - np.min(im)) / (np.max(im) - np.min(im)) # converts range to [0, 1]
            im = (2 * im) - 1

        # transpose everything but batch dimension
        targets = np.expand_dims(im[0].transpose(), 0)

        # set some more params
        print('writing to', f'projections/{image_prefix}')
        png_prefix = dnnlib.make_run_dir_path(f'projections/{image_prefix}_%04d-' % 0)
        self.proj.num_steps = num_steps
        self.proj.initial_learning_rate = lr
        
        # run projection
        snapshot_steps = set(self.proj.num_steps - np.linspace(0, self.proj.num_steps,
                                                          num_snapshots, endpoint=False, dtype=int))
        self.proj.start(targets)
        ims_list = []
        while self.proj.get_cur_step() < self.proj.num_steps:
            print('\r%d / %d ... ' % (self.proj.get_cur_step(), self.proj.num_steps), end='', flush=True)
            self.proj.step()
            if self.proj.get_cur_step() in snapshot_steps:
                misc.save_image_grid(self.proj.get_images(),
                                     png_prefix + f'step{self.proj.get_cur_step():04d}.png',
                                     drange=[-1, 1])
                im_latent = self.proj.get_images()
                ims_list.append(get_transformed_im(im_latent))
        print('\r%-30s\r' % '', end='', flush=True)

        
        
        return self.proj.get_dlatents(), ims_list
    
    
def get_transformed_im(im):
    im = np.transpose(im,
                  axes=(0, 3, 1, 2))
    return np.rot90(util.norm(im), k=-1,
                     axes=(1, 2))