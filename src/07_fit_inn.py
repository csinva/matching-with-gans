
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
import datetime
import time
import jsonlines
from PIL import Image
from scipy.optimize import curve_fit
import scipy as sy
import pickle as pkl
import seaborn as sns
from os.path import join as oj
import sklearn.model_selection
import sklearn.linear_model
import numpy.linalg as npl
import pandas as pd
import viz
from tqdm import tqdm
from scipy import optimize
import scipy.stats
import torch
import models
import util
import style
from util import detach
import losses
import figs

PROCESSED_DIR = 'processed'
RESULTS_DIR = 'results'
DIRECTIONS_DIR = '../data/annotation-dataset-stylegan2/linear_models/new' # path to many datasets, includes on directory before the below dirs
GENERATING_LATENTS_DIR = '../data/annotation-dataset-stylegan2/data'
NUM_SWEEPS = 20
reg_params = np.logspace(-5, 5, num=NUM_SWEEPS) # increase penalization on corrs
fname = '07_df_loss_tradeoff_nonlinear_INN_3lay'
p = {
    'num_layers': 3,
    'hidden_size': 100,
    'optimizer': 'adam',
    'lr': 5e-4, # 1e-2 seems good for sgd, 5e-4 seems good for adam
    'EPOCHS_PER_RUN': 1200,
}
p['EPOCHS'] = p['EPOCHS_PER_RUN'] * (NUM_SWEEPS - 1)
device = 'cuda'

# load data
latents = np.load(oj(GENERATING_LATENTS_DIR, 'W.npy'))
annotations_dict = pkl.load(open(oj(PROCESSED_DIR, '01_annotations_dict.pkl'), 'rb'))
annotations_dict_names = pkl.load(open(oj(PROCESSED_DIR, '01_annotations_labels_dict.pkl'), 'rb'))

# rename keys
annotations_dict = {k.replace('calibration-random-v2-', '').replace('-000', ''): annotations_dict[k]
                    for k in annotations_dict}
attrs = ['age', 'facial-hair', 'skin-color', 'gender', 'hair-length', 'makeup']
N_A = len(attrs)

attr_mat = np.array([annotations_dict[attr].mean(axis=1) for attr in attrs]).transpose()
attr_mat = (attr_mat - attr_mat.mean(axis=0)) / attr_mat.std(axis=0)
X = latents
Y = np.zeros(latents.shape) # pad Y with zeros, only first N rows have attributes
Y[:, :N_A] = attr_mat
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.3, random_state=42)


# setup
torch.manual_seed(0)
m = models.get_INN(num_layers=p['num_layers'], hidden_size=p['hidden_size'],
                   input_size=X_train.shape[1]).to(device)
# m = models.LinearNet(num_layers=1, input_size=X_train.shape[1], output_size=y_train.shape[1]) # linear reg
# m = models.LinearNet(num_layers=3, input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_size=100)
X_train_t = torch.Tensor(X_train).to(device)
y_train_t_full = torch.Tensor(y_train).to(device)
y_train_t =y_train_t_full[:, :N_A]
X_test_t = torch.Tensor(X_test).to(device)
y_test_t_full = torch.Tensor(y_test).to(device)
y_test_t = y_test_t_full[:, :N_A]
try:
    m(X_train_t)
except:
    print('shape wrong')

if p['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(m.parameters(), lr=p['lr'])
else:
    optimizer = torch.optim.Adam(m.parameters(), lr=p['lr'])

# fit
class s:
    epochs = []
    mse = []
    spearman = []
    indep_corr = []
    mse_test = []
    spearman_test = []
    indep_corr_test = []
    reg_param = []
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(s).items()
                 if not attr.startswith('_')}
    
# train
i = 0
for epoch in range(p['EPOCHS']): 
    y_pred_full = m(X_train_t)
    y_pred = y_pred_full[:, :N_A]
    y_pred_test_full = m(X_test_t)
    y_pred_test = y_pred_test_full[:, :N_A]
    
    # remember thes are only looking at first N_A dims
    reg_param = reg_params[i]
    mse = losses.mse(y_pred, y_train_t) 
    corr = losses.calc_mean_corrs_between_attributes(y_pred)
    loss = mse + reg_param * corr
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 

    if epoch % p['EPOCHS_PER_RUN'] == 0:
        print(f'\nepoch {epoch}, mse {mse.item():.3E}, corr {corr.item():.3E}, reg_param {reg_param:.2E}')
        
        # params
        s.epochs.append(epoch)
        s.reg_param.append(reg_param)
        
        # training
        s.mse.append(detach(torch.mean(torch.square(y_pred - y_train_t))))
        s.spearman.append(util.spearman_mean(y_pred, y_train_t))
        s.indep_corr.append(detach(losses.calc_mean_corrs_between_attributes(y_pred)))
        
        # testing
        s.mse_test.append(detach(torch.mean(torch.square(y_pred_test - y_test_t))))
        s.spearman_test.append(util.spearman_mean(y_pred_test, y_test_t))
        s.indep_corr_test.append(detach(losses.calc_mean_corrs_between_attributes(y_pred_test)))
        i += 1
    if epoch % 100 == 0:
        print(f'\tmse {mse.item():.3E}, corr {corr.item():.3E}')        

# save results as dataframe
s_dict = s._dict(s)
df = pd.DataFrame.from_dict(s_dict)
df.to_pickle(oj(PROCESSED_DIR, fname + '.pkl'))
torch.save(m.state_dict(), oj(PROCESSED_DIR, fname + '.weights'))
pkl.dump(p, open(oj(PROCESSED_DIR, fname + '_params.pkl'), 'wb'))