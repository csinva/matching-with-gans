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
    
    def gen(self, z):
        return self.Gs.run(z, None, **self.Gs_kwargs) # [minibatch, height, width, channel]
    
    
    def project(self, im: np.ndarray, image_prefix='im0', num_steps=50):
        '''Projects an image into the latent space
        
        Params
        ------
        im
            (batch_dim, H, W, channels)
            
        Returns
        -------
        latents
            (batch_dim, 18, 512)
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
        num_snapshots = 10
        self.proj.num_steps = num_steps
        
        # run projection
        snapshot_steps = set(self.proj.num_steps - np.linspace(0, self.proj.num_steps,
                                                          num_snapshots, endpoint=False, dtype=int))
        self.proj.start(targets)
        
        while self.proj.get_cur_step() < self.proj.num_steps:
            print('\r%d / %d ... ' % (self.proj.get_cur_step(), self.proj.num_steps), end='', flush=True)
            self.proj.step()
            if self.proj.get_cur_step() in snapshot_steps:
                misc.save_image_grid(self.proj.get_images(),
                                     png_prefix + f'step{self.proj.get_cur_step():04d}.png',
                                     drange=[-1, 1])
        print('\r%-30s\r' % '', end='', flush=True)

        return self.proj.get_dlatents()
    
    
    