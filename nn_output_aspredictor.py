# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:07:15 2019

@author: Tony Bigelow
"""

import pandas as pd
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
"""
bad_idcs = pd.isnull(df_now).any(1).nonzero()[0] #return all rows with a nan value
df_now = df_now.drop(df_now.index[bad_idcs])
im_now = np.delete(im_now,bad_idcs,axis=0)
"""
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
tstset = v4_dataset(df_now[:50],test_ims,t) # just dummy responses; only going to use the images

trainloader = DataLoader(dtset,batch_size=4)
testloader = DataLoader(tstset,batch_size=4)
#%% set up models

#returns a tuple with outputs at each layer in alexnet (minus fc layers)

class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        features = list(models.alexnet(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()
        
    def forward(self,x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x); #import pdb; pdb.set_trace()
            if ii in {0, 3, 6, 8, 10}: #only conv layers
                results.append(x)
        
        """
        an_outputs = namedtuple("AlexNetOutputs",['conv1','relu1','maxpool1',
                                                  'conv2','relu2','maxpool2',
                                                  'conv3','relu3','conv4',
                                                  'relu4','conv5','relu5','maxpool5'])
        """
        
        an_outputs = namedtuple("AlexnetOutputs",['conv1','conv2','conv3','conv4','conv5'])
        
        return an_outputs(*results)

#mods = [Lasso(alpha=1000000), Ridge(alpha=100000), ElasticNet(),
#          RandomForestRegressor(max_depth=7, n_estimators=100)] 
mods = [Lasso(alpha=1)]
#m_names = ['Lasso', 'Ridge','ElasticNet','RForest','ETrees']

#all_params = [{'alpha': (1e-2, 1e2)}, {'alpha': (1e-2, 1e2)}, {'alpha': (1e-2, 1e2)},
 #             {'max_depth': (3, 15)}]

all_params = [{'alpha': (1e-4, 1e2)}]
def train_models_fun(model, X_full, y_full):
    def test_model(**params):
        model.set_params(**params)
        scores = cross_val_score(model, X_full, y_full,
                                 cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=42),
                                 scoring=make_scorer(r2_score))
        r2_test = np.mean(scores)
        return r2_test
    return test_model


#%% Instantiate model, get output for training images
"""   
net=AlexNet() 
c1 = np.empty(shape=(1,64,55,55)); c2 = np.empty(shape=(1,192,27,27)); c3 = np.empty(shape=(1,384,13,13)); 
c4 = np.empty(shape=(1,256,13,13)); c5 = np.empty(shape=(1,256,13,13)); 
 
for i, data in enumerate(trainloader):
    
    inputs = data['image']
    
    outputs = net(inputs)
    
    c1=np.vstack((c1,outputs.conv1.detach().numpy()))
    c2=np.vstack((c2,outputs.conv2.detach().numpy()))
    c3=np.vstack((c3,outputs.conv3.detach().numpy()))
    c4=np.vstack((c4,outputs.conv4.detach().numpy()))
    c5=np.vstack((c5,outputs.conv5.detach().numpy()))
    
    
c1 = np.delete(c1,0,0); c2 = np.delete(c2,0,0); c3 = np.delete(c3,0,0); c4 = np.delete(c4,0,0); c5 = np.delete(c5,0,0);
c1 = c1[:,:,25:29,25:29]; c2 = c2[:,:,11:15, 11:15]; c3 = c3[:,:,5:9,5:9]; c4 = c4[:,:,5:9,5:9]; c5 = c5[:,:,5:9,5:9]; #grab just img center info
c1 = c1.reshape(551,-1); c2 = c2.reshape(551,-1); c3 = c3.reshape(551,-1); c4 = c4.reshape(551,-1); c5 = c5.reshape(551,-1); #reshape into n samples x nfeatures

conv_train = {'conv1': c1, 'conv2': c2, 'conv3': c3, 'conv4': c4, 'conv5': c5}


c1 = np.empty(shape=(1,64,55,55)); c2 = np.empty(shape=(1,192,27,27)); c3 = np.empty(shape=(1,384,13,13)); 
c4 = np.empty(shape=(1,256,13,13)); c5 = np.empty(shape=(1,256,13,13)); 

for i, data in enumerate(testloader): 
    
    inputs = data['image']
    
    outputs = net(inputs)
    
    c1=np.vstack((c1,outputs.conv1.detach().numpy()))
    c2=np.vstack((c2,outputs.conv2.detach().numpy()))
    c3=np.vstack((c3,outputs.conv3.detach().numpy()))
    c4=np.vstack((c4,outputs.conv4.detach().numpy()))
    c5=np.vstack((c5,outputs.conv5.detach().numpy()))
    
c1 = np.delete(c1,0,0); c2 = np.delete(c2,0,0); c3 = np.delete(c3,0,0); c4 = np.delete(c4,0,0); c5 = np.delete(c5,0,0);
c1 = c1[:,:,25:29,25:29]; c2 = c2[:,:,11:15, 11:15]; c3 = c3[:,:,5:9,5:9]; c4 = c4[:,:,5:9,5:9]; c5 = c5[:,:,5:9,5:9]; #grab just img center info
c1 = c1.reshape(50,-1); c2 = c2.reshape(50,-1); c3 = c3.reshape(50,-1); c4 = c4.reshape(50,-1); c5 = c5.reshape(50,-1); #reshape into n samples x nfeatures

conv_test = {'conv1': c1, 'conv2': c2, 'conv3': c3, 'conv4': c4, 'conv5': c5}

np.save('data/conv_train.npy',conv_train)
np.save('data/conv_test.npy',conv_test)
"""
#%% Now fit the data
best_r2 = 0
n_dict = {}

conv_train = np.load('data/conv_train.npy').flat[0]
conv_test = np.load('data/conv_test.npy').flat[0]

conv_train.pop('conv1',None)

for nnn in trange(1,df_now.shape[1],ncols=20):
    best_r2 = 0
    
    model_dict = {}
    for modelnum in range(len(mods)):
        
        model = mods[modelnum]
        model_params = all_params[modelnum]
        
        y_full = df_now.iloc[:,nnn]
        
        good = ~np.isnan(y_full)
        
        ytrain = np.array(y_full.loc[good])
        iii = 0
        for layer in conv_train.keys():
            xtrain = conv_train[layer][good,:]
            xtest = conv_test[layer]
            fun = train_models_fun(model,xtrain,ytrain)
            net_opt = BayesianOptimization(fun,model_params,verbose=1)
            net_opt.maximize(n_iter=100,acq="poi",xi=1e-1)
            
            try:
                r2_test = net_opt.max['target']
                best_params = net_opt.max['params']
                model.set_params(**best_params)
            except:
                r2_test = 0
                print('Val error')
            
            if r2_test > best_r2:
                model.fit(xtrain,ytrain)
                out = model.predict(xtest)
                test.iloc[:,nnn] = out
                best_r2 = r2_test

test.to_csv('data/output.csv',index=False)
