from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
from os.path import join as oj
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
from tqdm import tqdm
import argparse
import config

def predict_age_gender_race(OUT_DIR, imgs_path = 'cropped_faces/'):
    img_names = [os.path.join(imgs_path, x)
                 for x in sorted(os.listdir(imgs_path)) 
                 if '.jpg' in x] #[:20]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('fair_face_models/fairface_alldata_20191111.pt'))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load('fair_face_models/fairface_alldata_4race_20191111.pt'))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    for index, img_name in tqdm(enumerate(img_names)):
        if index % 1000 == 0:
            print("Predicting... {}/{}".format(index, len(img_names)))

        face_names.append(img_name)
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        outputs = model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)

        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    result = pd.DataFrame([face_names,
                           race_preds_fair,
                           race_preds_fair_4,
                           gender_preds_fair,
                           age_preds_fair,
                           race_scores_fair, race_scores_fair_4,
                           gender_scores_fair,
                           age_scores_fair, ]).T
    result.columns = ['face_name_align',
                      'race_preds_fair',
                      'race_preds_fair_4',
                      'gender_preds_fair',
                      'age_preds_fair',
                      'race_scores_fair',
                      'race_scores_fair_4',
                      'gender_scores_fair',
                      'age_scores_fair']
    result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
    result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
    result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
    result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
    result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
    result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
    result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

    # race fair 4

    result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
    result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
    result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
    result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

    # gender
    result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
    result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

    # age
    result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
    result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
    result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
    result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
    result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
    result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
    result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
    result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
    result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

    
    # save
    os.makedirs(OUT_DIR, exist_ok=True)
    result['img_names'] = img_names
    r = result[['face_name_align',
            'race', 'race4',
            'gender', 'age',
            'race_scores_fair', 'race_scores_fair_4',
            'gender_scores_fair', 'age_scores_fair', 'img_names']]
    r.to_csv(oj(OUT_DIR, 'race.csv'), index=False)
    r.to_pickle(oj(OUT_DIR, 'race.pkl'))

    print("saved results at ", OUT_DIR)



if __name__ == "__main__":
    
    print("using CUDA?: %s" % dlib.DLIB_USE_CUDA)
#     IM_DIR = DIR_IMS #'../../data/celeba-hq/ims/'
#     OUT_DIR = DIR_CELEBA #'../../data_processed/celeba-hq/attr_preds'
    print("loading faces from ", IM_DIR)
    predict_age_gender_race(config.DIR_IMS, config.DIR_CELEBA)