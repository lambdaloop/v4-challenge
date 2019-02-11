# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 23:01:41 2019

@author: Tony Bigelow
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:47:54 2019

@author: Tony Bigelow>
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
    


t = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
dtset = v4_dataset(df_now,im_now,t)

all_idx = np.arange(len(dtset)) #no. conditions w/ full data
trainidx,testidx = train_test_split(
        all_idx,test_size=0.2,random_state=42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_net(lr,mom):
    net = models.alexnet(pretrained=True)
    num_ft = net.classifier[6].in_features
    net.classifier[6] = nn.Linear(num_ft,18)
    net.to(device)
    
    for param in net.parameters():
        param.requires_grad=False
        
    for param in net.classifier[6].parameters():
        param.requires_grad = True
    
    
    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(net.parameters(),lr=lr,momentum=mom)
    
    for epoch in range(1):
        
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
            
            responses = torch.from_numpy(responses);

            inputs, responses = inputs.to(device), responses.to(device)
            output = net(inputs.unsqueeze(0))
            
            #var.append(np.corrcoef(responses.cpu().numpy(),output.cpu().numpy())[0,1]**2)
            var.append(r2_score(responses.cpu().numpy().reshape(-1,1),output.cpu().numpy().reshape(-1,1),multioutput='variance_weighted'))
    var = array(var)
    print('Mean: '+str(np.mean(var)))
    return np.mean(var)


net_opt= BayesianOptimization(train_net,{'lr':(1e-4,1e2), 'mom':(0.5,0.99)},verbose=True)
net_opt.maximize(n_iter=30,acq="poi",xi=1e-1)

best_params = net_opt.max['params']

import pdb; pdb.set_trace()

net = models.alexnet(pretrained=True)
num_ft = net.classifier[6].in_features
net.classifier[6] = nn.Linear(num_ft,18)

for param in net.parameters():
    param.requires_grad=False
    
for param in net.classifier[6].parameters():
    param.requires_grad = True


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),lr=best_params['lr'],momentum=best_params['mom'])

for epoch in range(1):
    
    for idx in trainidx:
        inputs, responses = dtset[idx]['image'],dtset[idx]['responses']
        #import pdb; pdb.set_trace()
        optimizer.zero_grad()        
        
        responses = torch.from_numpy(responses); responses.requires_grad=True
        
        outputs = net(inputs.unsqueeze(0))
        
        loss = criterion(outputs.data.double(),responses)
        loss.backward()
        optimizer.step()
            
net.eval()
var = []
with torch.no_grad():
    for idx in testidx:
        inputs, responses = dtset[idx]['image'],dtset[idx]['responses']
        
        responses = torch.from_numpy(responses);

        #inputs, responses = inputs.to(device), responses.to(device)
        output = net(inputs.unsqueeze(0))
        
        var.append(np.corrcoef(responses.numpy(),output.numpy())[0,1]**2)
var = array(var)
print('Mean: '+str(np.mean(var)))

    
for idx in range(len(test_ims)):

    input_now = Image.fromarray(np.uint8(test_ims[idx,:,:,:]*255))
    input_now = t(input_now)
    output = net(input_now.unsqueeze(0)); output = output.numpy()
	
    test.iloc[idx,1:] = output.T.squeeze(1) #change from 18,1 to 18, 
        
        
test.to_csv('data/output.csv', index=False)
