# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:47:54 2019

@author: Tony Bigelow
"""
#%%

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

os.chdir(r'C:\Users\Tony Bigelow\Desktop\Hackathon\v4-challenge')

warnings.simplefilter(action='ignore',category=FutureWarning)

lasso_model = Lasso(alpha=1)
l_params = {'alpha': (1e-2,1e2)}


#%%
class v4_dataset(Dataset):
    
    def __init__(self,transform):
        
        self.resp_frame = pd.read_csv('./data/train.csv') 
        self.resp_frame = self.resp_frame.dropna()
        self.transform = transform
        self.ims = np.load('./data/stim.npy')
        
    def __len__(self):
        return len(self.resp_frame)
    
    def __getitem__(self,idx):
        img = Image.fromarray(np.uint8(self.ims[idx,:,:,:]*255))
        responses = self.resp_frame.iloc[idx,1:].as_matrix()
        
        if self.transform is not None:
            img = self.transform(img)
            
        sample = {'image': img, 'responses': responses}
        return sample
    


t = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
dtset = v4_dataset(t)

all_idx = np.arange(len(dtset)) #no. conditions w/ full data
trainidx,testidx = train_test_split(
        all_idx,test_size=0.1,random_state=42)

device = torch.device("cuda:0" if torch.cuda_is_availble() else "cpu")

def train_net(lr,mom):
    net = models.alexnet(pretrained=True)
    num_ft = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ft,18)

    for param in net.parameters():
        param.requires_grad=False
        
    for param in net.classifier[6].parameters():
        param.requires_grad = True
    
    net.to(device)
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum=mom)
    
    for epoch in range(5):
        
        for idx in trainidx:
            inputs, responses = dtset[idx]['image'],dtset[idx]['responses']
            #import pdb; pdb.set_trace()
            optimizer.zero_grad()
            
            responses = torch.from_numpy(responses); responses.requires_grad=True
            inputs, responses = inputs.to(device), responses.to(device)
            
            outputs = net(inputs.unsqueeze(0))
            
            loss = criterion(outputs.data.double(),responses)
            loss.backward()
            optimizer.step()
            
    net.eval()
    var = []
    with torch.no_grad():
        for idx in testidx:
            inputs, responses = dtset[idx]['image'],dtset[idx]['responses']
            
            inputs, responses = inputs.to(device), responses.to(device)
            output = net(inputs.unsqueeze(0))
            
            var.append(np.corrcoef(responses,output.data.numpy())[0,1]**2)
    var = array(var)
    
    return np.mean(var)


net_opt= BayesianOptimization(train_net,{'lr':(1e-6,1e-2), 'mom':(0.3,0.99)},verbose=1)
net_opt.maximize(n_iter=100)

print(net_opt.res['max'])