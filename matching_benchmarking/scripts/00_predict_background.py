import sys

from os.path import join as oj

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from matching import *
from tqdm import tqdm

import config
import data
import util

sys.path.append(config.LIB_PATH)
from face_segmentation.models import LinkNet34
import config
from config import *

if __name__ == '__main__':
    # df contains filenames, ids, and attributes
    df = data.load_all_labs()
    df = df.set_index('fname_id')

    # get face seg model
    device = torch.device('cpu')  # "cuda:0" if torch.cuda.is_available() else "cpu")
    model = LinkNet34()
    model.load_state_dict(torch.load(oj(config.LIB_PATH, 'face_segmentation/linknet.pth'),
                                     map_location=lambda storage, loc: storage))
    model.eval()
    model.to(device)

    # setup for face segmentation
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    ])
    t = transforms.Resize(size=(256, 256))


    def smooth_mask(mask):
        im = Image.fromarray(mask)
        im2 = im.filter(ImageFilter.MinFilter(3))
        im3 = im2.filter(ImageFilter.MaxFilter(5))
        return np.array(im3)


    def get_mask(path, plot=True):
        _img = Image.open(path)
        a = img_transform(_img)
        a = a.unsqueeze(0)
        imgs = a.to(device)
        pred = model(imgs)
        img = np.array(_img)
        mask = pred > 0.5
        mask = mask.squeeze()
        mask = mask.cpu().numpy()
        img = np.array(t(Image.fromarray(img)))
        if plot:
            img[mask == 0] = 170
            util.imshow(img)
        return mask, img


    # test on one image
    '''
    for i in range(3):
        fname = oj(DIR_IMS, df.iloc[i].fname_final)
        file_path = oj(fname)
        model.eval()
        mask, img = get_mask(file_path) # mask is True for face, false otherwise
        plt.show()
    '''

    # loop over everything and run
    r = {
        k: [] for k in ['mean', 'std']
    }
    for i in tqdm(range(df.shape[0])):
        fname = oj(DIR_IMS, df.iloc[i].fname_final)
        mask, img = get_mask(fname, plot=False)  # mask is True for face, false otherwise
        background = img[~mask]
        r['mean'].append(background.mean())
        r['std'].append(background.std())

    pd.DataFrame(r).to_pickle(oj(DIR_PROCESSED, 'background_stats.pkl'))
