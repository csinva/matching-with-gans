import os
from copy import deepcopy
from os.path import join as oj

import h5py
import numpy as np
import sklearn.metrics
import torch
import torchvision
from tqdm import tqdm
from config import *

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
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def encode(self, x):
        x = (x - self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(112, 112), align_corners=False)
        # x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        '''
        encodings = []
        for block in self.blocks:
            x = block(x)
            encodings.append(deepcopy(x.cpu().detach().numpy().flatten()))
        return np.hstack(encodings).flatten()
        '''
        x = self.blocks[0](x)
        return deepcopy(x.cpu().detach().numpy().flatten())
        # print(encodings[0].shape, encodings[1].shape, encodings[2].shape, encodings[1].shape)

    def forward(self, encoding1, encoding2):
        return torch.nn.functional.l1_loss(encoding1, encoding2)


if __name__ == '__main__':
    DIR_ENCODINGS = '../data_processed/celeba-hq/encodings_vgg_small/'
    out_fname = 'processed/13_facial_dists_pairwise_vgg_small.h5'
    os.makedirs(DIR_ENCODINGS, exist_ok=True)

    # get fnames
    fnames = sorted([f for f in os.listdir(DIR_IMS) if '.jpg' in f])
    n = len(fnames)

    # calc encodings
    '''
    m = VGGPerceptualLoss()
    for i in tqdm(range(n)):
        fname_out = oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy'
        if not os.path.exists(fname_out):
            image = Image.open(oj(DIR_IMS, fnames[i]))
            x = TF.to_tensor(image)
            x.unsqueeze_(0)
            encoding = m.encode(x)
            np.save(open(fname_out, 'wb'), encoding)
    
    
    # initial write
    print('initial write...')
    f = h5py.File(out_fname, 'w')
    dset = f.create_dataset("dists", (n, n), dtype='f')
    dset[:] = 0
    '''

    # append
    dset = h5py.File(out_fname, 'a')['dists']

    print('loading encodings...')
    encoding_fnames = [oj(DIR_ENCODINGS, fnames[i][:-4]) + '.npy' for i in range(n)]
    BLOCK_SIZE = 100
    blocks_computed = 0
    total_blocks = int((n // BLOCK_SIZE) ** 2 / 2)
    for block_num_i in tqdm(range(n // BLOCK_SIZE)):
        encodings_i = np.vstack([np.load(open(f, 'rb'))
                                 for f in encoding_fnames[block_num_i * BLOCK_SIZE: (block_num_i + 1) * BLOCK_SIZE]])
        for block_num_j in range(block_num_i + 1):
            encodings_j = np.vstack([np.load(open(f, 'rb'))
                                     for f in
                                     encoding_fnames[block_num_j * BLOCK_SIZE: (block_num_j + 1) * BLOCK_SIZE]])

            # check if this block has been written
            blocks_computed += 1
            if dset[block_num_i * BLOCK_SIZE, block_num_j * BLOCK_SIZE] == 0:
                # print(encodings_i.shape, encodings_i[0].shape, encodings_j.shape)
                facial_dists = sklearn.metrics.pairwise_distances(encodings_i, encodings_j, metric='l1', n_jobs=4)
                dset[block_num_i * BLOCK_SIZE: (block_num_i + 1) * BLOCK_SIZE,
                block_num_j * BLOCK_SIZE: (block_num_j + 1) * BLOCK_SIZE] = deepcopy(facial_dists)
                # print(blocks_computed, '/', total_blocks, 'computed')
            # else:
            # print(blocks_computed, '/', total_blocks, 'skipped')

    # still need to copy top-right to bot-left
    dset[np.eye(n).astype(bool)] = 1e3  # don't pick same point
    for i in range(n):
        for j in range(i):
            dset[j, i] = dset[i, j]
