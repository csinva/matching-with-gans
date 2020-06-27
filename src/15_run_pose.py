import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import torch.nn.functional as F
import sys
sys.path.append('../lib/deep_head_pose/code/')
import hopenet, utils # datasets, hopelessnet
from PIL import Image
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    image_path = "/home/ubuntu/face-disentanglement/data/celeba-hq/ims/"
    fname_out = '/home/ubuntu/face-disentanglement/data_processed/celeba-hq/pose.pkl'
    device = 'cpu'
    snapshot_path = '/home/ubuntu/face-disentanglement/lib/deep_head_pose/models/hopenet_resnet18.pkl'
    
    print('Loading snapshot.')
    model = hopenet.Hopenet(
        torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model = model.eval().to(device)

    print('Loading data.')
    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    
    fnames = sorted([fname for fname in os.listdir(image_path)
                     if 'jpg' in fname])
    
    yaws = []
    pitches = []
    rolls = []
    for fname in tqdm(fnames):
        fname_full = os.path.join(image_path, fname)
        image = Image.open(fname_full)
        im_t = transformations(image).unsqueeze(0)
        # print(model(im_t))
        yaw, pitch, roll = model(im_t)
        yaws.append(yaw.detach().cpu().numpy())
        pitches.append(pitch.detach().cpu().numpy())
        rolls.append(roll.detach().cpu().numpy())

    pd.DataFrame.from_dict({
        'yaws': yaws,
        'pitches': pitches,
        'rolls': rolls
    }).to_pickle(fname_out)
