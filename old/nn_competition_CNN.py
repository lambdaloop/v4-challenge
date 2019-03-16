# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:56:47 2019

@author: tony
"""
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
#import os
#os.chdir('/home/tony/NN_competition')
import os
os.chdir(r"C:\Users\Tony Bigelow\Desktop\Hackathon\NN kaggle comp Dean")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#%% Let's load some images and and look at their 
# responses

stim_list = np.load('stim.npy')


class rf():
    def __init__(self,csv_file):
        
        self.resp_frame = pd.read_csv(csv_file)
        
    def __getitem__(self,idx):
        
        return self.resp_frame.iloc[idx,:].as_matrix()

train_resps = rf('train.csv')

#%% Calculate statistical features

def calc_stats(image):
    
    results = []
    
    for j in range(3):
        results.append(np.mean(image[:,:,j]))
        results.append(np.std(image[:,:,j]))
        results.append(kurtosis(image[:,:,j],axis=None))
        results.append(skew(image[:,:,j],axis=None))
        im = image[20:60,20:60,j]
        results.append(np.min(im))
        results.append(np.max(im))
        results.append(np.sqrt(np.mean(image[:,:,j]**2)))
        
    return np.array(results)


#%% 

