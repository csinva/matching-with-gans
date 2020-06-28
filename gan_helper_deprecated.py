import sys

from config import *

sys.path.append(DIR_STYLEGAN)
import project_images
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks
import projector
import numpy as np
import util


class Generator:
    def __init__(self, image_size=512, truncation_psi=0.5):
        network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'

        # load the networks
        tflib.init_tf()
        print('Loading networks from "%s"...' % network_pkl)
        _, _, Gs = pretrained_networks.load_networks(network_pkl)
        Gs_kwargs = dnnlib.EasyDict()
        Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            Gs_kwargs.truncation_psi = truncation_psi

        # store generator
        self.Gs = Gs
        self.Gs_kwargs = Gs_kwargs
        self.image_size = image_size
        self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

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
        return self.Gs.components.mapping.run(z, None,
                                              truncation_psi=0.5,
                                              randomize_noise=False)  # randomize_noise=False) # [minibatch, layer, component]

    def gen(self, z):
        '''Generate image from shared latent
        Params
        ------
        z: np.ndarray
            (batch_size, 512)
        '''
        return self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]

    def gen_full(self, z_full):
        '''Generate image from full latent
        Params
        ------
        z: np.ndarray
            (batch_size, 18, 512)
        '''
        im = self.Gs.components.synthesis.run(z_full, **self.Gs_kwargs)
        return im

    def gen_full_old(self, z_full):
        '''Generate image from full latent
        Params
        ------
        z: np.ndarray
            (batch_size, 18, 512)
        '''
        im = tflib.run(self.Gs.components.synthesis.get_output_for(z_full))
        return get_transformed_im(im)
    
####################################### Redundant functions ###################    
    
    def generateImageFromStyle(self, w):
        '''
        w
            (N, 512)
        '''
        w = np.tile(np.expand_dims(w, 1), (1, 18, 1))
        img = self.Gs.components.synthesis.run(w, is_validation=True, randomize_noise=False, 
                                               output_transform=self.fmt)
        return self.processImage(img)
    
    def generateImageFromStyleFull(self, w):
        '''
        w
            (N, 18, 512)
        '''
        img = self.Gs.components.synthesis.run(w, is_validation=True, randomize_noise=False, 
                                               output_transform=self.fmt)
        return self.processImage(img)    

    def generateImageFromLatents(self, z):
        img = self.Gs.run(z, None, is_validation=True, randomize_noise=False, 
                          output_transform=self.fmt)
        return self.processImage(img)


    def processImage(self, img):
        img_resize = np.zeros((img.shape[0], self.image_size, self.image_size, 3))
        for j in range(img.shape[0]):
            img_resize[j, ...] = cv2.resize(img[j, ...], (self.image_size, self.image_size))
        img = img_resize/255.0
        return img

####################################### PROJECTION ###################
    
    def initialize_projector(self,
                             vgg16_pkl='https://drive.google.com/uc?id=1N2-m9qszOeVC9Tq77WxsLnuWwOedQiD2',
                             num_steps=1000,
                             initial_learning_rate=0.1,
                             initial_noise_factor=0.05,
                             verbose=False,
                             regularize_mean_deviation_weight=0.1):
        '''Sets self.proj
        '''
        self.proj = projector.Projector(
            vgg16_pkl=vgg16_pkl,
            num_steps=num_steps,
            initial_learning_rate=initial_learning_rate,
            initial_noise_factor=initial_noise_factor,
            verbose=verbose,
            regularize_mean_deviation_weight=regularize_mean_deviation_weight
        )
        self.proj.set_network(self.Gs)

    def project(self, fname):
        '''Project an image given a filename (assumes projector was initialized)
        Params
        ------
        fname: str
            path to the image to be projected
        
        Returns
        -------
        latents
            (18, 512)
        image: np.ndarray
            (1024, 1024, 3)
            
        also saves a little bit to 'tmp' directory
        '''
        latents, im_rec = project_images.project_image(self.proj, src_file=fname, dst_dir=None, tmp_dir='tmp',
                                                       video=False)
        return latents, im_rec


def get_transformed_im(im):
    im = np.transpose(im,
                      axes=(0, 3, 1, 2))
    return np.rot90(util.norm(im), k=-1,
                    axes=(1, 2))
