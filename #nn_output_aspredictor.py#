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
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics.scorer import make_scorer
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import pandas as pd

#os.chdir(r'C:\Users\Tony Bigelow\Desktop\Hackathon\v4-challenge')

warnings.simplefilter(action='ignore',category=FutureWarning)

lasso_model = Lasso(alpha=1)
l_params = {'alpha': (1e-2,1e2)}

df_now = pd.read_csv('./data/train.csv')
im_now = np.load('./data/stim.npy')
test_ims = im_now[:50,...]
im_now = im_now[50:,...]#eliminate test images for training

bad_idcs = pd.isnull(df_now).any(1).nonzero()[0] #return all rows with a nan value
df_now = df_now.drop(df_now.index[bad_idcs])
im_now = np.delete(im_now,bad_idcs,axis=0)

test = pd.read_csv('data/sub.csv')

#%%
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
net = models.alexnet(pretrained=True)

dtset = v4_dataset(df_now, im_now,t)

for i in range(len(net.features)):
    
    