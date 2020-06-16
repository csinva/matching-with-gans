import numpy as np
import cv2
import sys
sys.path.append('stylegan2')
import dnnlib
import dnnlib.tflib as tflib
import pretrained_networks


class GANWrapper(object):
    def __init__(self, image_size=512):

        tflib.init_tf()
        network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
#         path = './stylegan2/cache/stylegan2-ffhq-config-f.pkl'
        _, _, Gs = pretrained_networks.load_networks(network_pkl)

        self.Gs = Gs
        self.image_size = image_size
        self.fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


    def getStyle(self, z):
        return self.Gs.components.mapping.run(z, None, is_validation=True)[:, 0, :]


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
