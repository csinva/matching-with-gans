import sys
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from os.path import join as oj
import pandas as pd
import pickle as pkl
import models
import util
import os
import config
import viz
import scipy.stats
from tqdm import tqdm
import figs
import matplotlib.image as mpimg
import seaborn as sns
import data
import torch
import torchvision
from copy import deepcopy
from PIL import Image
import torchvision.transforms.functional as TF

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
    
    def encode(self, x):
        x = (x - self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        encodings = []
        for block in self.blocks:
            x = block(x)
            encodings.append(deepcopy(x.cpu().detach().numpy().flatten()))
        # print(encodings[0].shape, encodings[1].shape, encodings[2].shape)
        return np.hstack(encodings).flatten()
        
    def forward(self, encoding1, encoding2):
        return torch.nn.functional.l1_loss(encoding1, encoding2)



if __name__ == '__main__':
    DIR_ORIG = '../data/celeba-hq/ims/'
    DIR_ENCODINGS = '../data_processed/celeba-hq/encodings_vgg/'
    out_fname = 'processed/13_facial_dists_pairwise_vgg.npy'
    os.makedirs(DIR_ENCODINGS, exist_ok=True)
    
    # get fnames
    fnames = sorted([f for f in os.listdir(DIR_ORIG) if '.jpg' in f])
    n = len(fnames)
    
    
    # calc encodings
    m = VGGPerceptualLoss()
    for i in tqdm(range(n)):
        fname_out = oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy'
        if not os.path.exists(fname_out):
            image = Image.open(oj(DIR_ORIG, fnames[i]))
            x = TF.to_tensor(image)
            x.unsqueeze_(0)
            encoding = m.encode(x)
#             im = mpimg.imread(oj(DIR_ORIG, fnames[i]))
#             encoding = face_recognition.face_encodings(im, model='cnn')
#             if len(encoding) > 0:
#                 encoding = encoding[0]
            np.save(open(fname_out, 'wb'), encoding)
    

    # calc failures
    FAILURES_FILE = oj(DIR_ENCODINGS, 'failures_vgg.npy')
    if not os.path.exists(FAILURES_FILE):
        failures = []
        for i in tqdm(range(n)):
            fname_out = oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy'
            encoding = np.load(open(fname_out, 'rb'))
            if not np.any(encoding):
                failures.append(i)
        np.save(open(FAILURES_FILE, 'wb'), np.array(failures))
    else:
        failures = np.load(open(FAILURES_FILE, 'rb'))
    
    # calc dists
    def l1_loss(x1, x2):
        return np.sum(np.abs(x1 - x2))
    
    dists_facial = np.ones((n, n)) * 1e3
    print('loading encodings...')
    encodings_fnames = [oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy' for i in range(n)]
    encodings = [oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy', 'rb'))
                 for i in range(n)]
    for i in tqdm(range(n)):
        if i in failures:
            continue
        encoding1 = np.load(open(encodings_fnames[i], 'rb'))
        for j in range(i):
            if j in failures:
                continue
            encoding2 = np.load(open(encodings_fnames[j], 'rb'))
            facial_dist = l1_loss(encoding1, encoding2)
            dists_facial[i, j] = facial_dist
            dists_facial[j, i] = facial_dist
    dists_facial[np.eye(n).astype(bool)] = 1e3 # don't pick same point
    np.save(open(out_fname, 'wb'), dists_facial)
    # pkl.dump({'facial_dists': dists_facial, 'ids': fname_ids}, open(out_fname, 'wb'))
    
    