# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:07:15 2019

@author: Tony Bigelow
"""

import pandas as pd

"""
aaa = pd.read_pickle('./model_perf_and_params.pkl')

"""
#%%
import warnings 
import numpy as np
from numpy import array 
from bayes_opt import BayesianOptimization
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split,cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics.scorer import make_scorer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import namedtuple
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from tqdm import tqdm, trange

#os.chdir(r'C:\Users\Tony Bigelow\Desktop\Hackathon\v4-challenge')

warnings.simplefilter(action='ignore',category=FutureWarning)

#%% organize data

df_now = pd.read_csv('./data/train.csv')
im_now = np.load('./data/stim.npy')
test_ims = im_now[:50,...]
im_now = im_now[50:,...]#eliminate test images for training

bad_idcs = pd.isnull(df_now).any(1).nonzero()[0] #return all rows with a nan value
df_now = df_now.drop(df_now.index[bad_idcs])
im_now = np.delete(im_now,bad_idcs,axis=0)

test = pd.read_csv('data/sub.csv')

class v4_dataset(Dataset):
    
    def __init__(self,df,ims,transform):
        
        self.resp_frame = df
        self.transform = transform
        self.ims = ims
        
    def __len__(self):
        return len(self.resp_frame)
    
    def __getitem__(self,idx):
        img = Image.fromarray(np.uint8(self.ims[idx,:,:,:]*255))
        responses = self.resp_frame.iloc[idx,1:].as_matrix()
        
        if self.transform is not None:
            img = self.transform(img)
            
        sample = {'image': img, 'responses': responses}
        return sample

t = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

dtset = v4_dataset(df_now, im_now,t)
#%% set up models

#returns a tuple with outputs at each layer in alexnet (minus fc layers)

class Alexnet(torch.nn.Module):
    def __init__(self):
        super(Alexnet,self).__init__()
        features = list(models.alexnet(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
        
    def forward(self,x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            results.append(x)
        an_outputs = namedtuple("AlexNetOutputs",['conv1','relu1','maxpool1',
                                                  'conv2','relu2','maxpool2',
                                                  'conv3','relu3','conv4',
                                                  'relu4','conv5','relu5','maxpool5'])
        return an_outputs(*results)

mods = [Lasso(alpha=1000000), Ridge(alpha=100000), ElasticNet(),
          RandomForestRegressor(max_depth=7, n_estimators=100),
          ExtraTreesRegressor(max_depth=7, n_estimators=100)] 
m_names = ['Lasso', 'Ridge','ElasticNet','RForest','ETrees']
all_params = [{'alpha': (1e-2, 1e2)}, {'alpha': (1e-2, 1e2)}, {'alpha': (1e-2, 1e2)},
              {'max_depth': (3, 15)}, {'max_depth': (3, 15)}]

def train_models_fun(model, X_full, y_full):
    def test_model(**params):
        model.set_params(**params)
        scores = cross_val_score(model, X_full, y_full,
                                 cv=ShuffleSplit(n_splits=1, test_size=0.1, random_state=42),
                                 scoring=make_scorer(r2_score))
        r2_test = np.mean(scores)
        return r2_test
    return test_model


#%% Instantiate model, get output for images
    
net=Alexnet() 
ft_vec = np.array([])

for x in dtset:
    
    outputs = net(x['image'].unsqueeze(0))
    
    ft_vec = np.append(ft_vec,outputs[-2].detach().numpy().squeeze(0))
#%% Now fit the data
n_dict = {}
for nnn in trange(1,df_now.shape[1],ncols=20):
    best_r2 = 0
    
    model_dict = {}
    for modelnum in range(len(mods),ncol=20):
        
        model = mods[modelnum]
        model_params = all_params[modelnum]
        
