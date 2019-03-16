# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:07:15 2019

@author: Tony Bigelow
"""

import pandas as pd
import warnings
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics.scorer import make_scorer
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import namedtuple
from tqdm import trange,tqdm

#os.chdir(r'C:\Users\Tony Bigelow\Desktop\Hackathon\v4-challenge')

warnings.simplefilter(action='ignore',category=FutureWarning)

#%% organize data

df_now = pd.read_csv('./data/train.csv')
im_now = np.load('./data/stim.npy')
im_mean = np.mean(np.mean(im_now,axis=(1,2)),axis=0)
im_std = np.std(np.std(im_now,axis=(1,2)),axis=0)
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

t = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=im_mean, std= im_std)])

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

class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        features = list(models.vgg16(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        out_dict = []; count = 1
        for ii, model in enumerate(self.features):
            x = model(x); #import pdb; pdb.set_trace()
            if type(model) == nn.modules.conv.Conv2d and ii >7: #only conv layers
                out_dict.append('conv'+str(count)); count += 1
                results.append(x)

        """
        an_outputs = namedtuple("AlexNetOutputs",['conv1','relu1','maxpool1',
                                                  'conv2','relu2','maxpool2',
                                                  'conv3','relu3','conv4',
                                                  'relu4','conv5','relu5','maxpool5'])
        """

        an_outputs = namedtuple("VGGOutputs",out_dict)

        return an_outputs(*results)

#mods = [Lasso(alpha=1000000), Ridge(alpha=100000), ElasticNet(),
#          RandomForestRegressor(max_depth=7, n_estimators=100)]
mods = [ElasticNet(alpha=1,l1_ratio=0.5)]
#m_names = ['Lasso', 'Ridge','ElasticNet','RForest','ETrees']

all_params = [{'alpha': (1e-2, 1e2), 'l1_ratio': (0,1)}]

#all_params = [{'alpha':(1e-4,1e3), 'l1_ratio':(0,1)},{'max_depth': (5,30), 'n_estimators': (20,400)}]
def train_models_fun(model, X_full, y_full):
    def test_model(**params):
        model.set_params(**params)
        scores = cross_val_score(model, X_full, y_full,
                                 cv=ShuffleSplit(n_splits=1, test_size=0.10, random_state=42),
                                 scoring=make_scorer(r2_score))
        r2_now = np.mean(scores)
        if r2_now < 0:
            r2_now =0

        return np.sqrt(r2_now)

    return test_model


#%% Instantiate model, get output for training imagesj


net = VGG()
net.eval()

cresps = dict()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

conv_train = dict()

for i, data in enumerate(trainloader):

    inputs = data['image']
    inputs = inputs.to(device)
    outputs = net(inputs)

    if i == 0:
        for j, fld in enumerate(outputs._fields):
            conv_train[fld] = outputs[j].cpu().detach().numpy()
    else:
        for j, fld in enumerate(outputs._fields):
            conv_train[fld] = np.vstack((conv_train[fld],outputs[j].cpu().detach().numpy()))

    print('{}% done with train examples'.format((i+1)*100/len(trainloader)))

conv_test = dict()

for i, data in enumerate(testloader):

    inputs = data['image']
    inputs = inputs.to(device)
    outputs = net(inputs)

    if i == 0:
        for j, fld in enumerate(outputs._fields):
            conv_test[fld] = outputs[j].cpu().detach().numpy()

    else:
        for j, fld in enumerate(outputs._fields):
            conv_test[fld] = np.vstack((conv_test[fld],outputs[j].cpu().detach().numpy()))


    print('{}% done with test examples'.format((i+1)*100/len(testloader)))


"""
#net=AlexNet()
net = models.resnet18(pretrained=True)
res18_conv = nn.Sequential(*list(net.children())[:-2])

for param in res18_conv.parameters():
    param.requires_grad=False


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

c1 = np.empty(shape=(1,64,55,55)); c2 = np.empty(shape=(1,192,27,27)); c3 = np.empty(shape=(1,384,13,13));
c4 = np.empty(shape=(1,256,13,13)); c5 = np.empty(shape=(1,256,13,13));

for i, data in enumerate(trainloader):

    inputs = data['image']

    inputs = inputs.to(device)

    outputs = net(inputs)

    c1=np.vstack((c1,outputs.conv1.cpu().detach().numpy()))
    c2=np.vstack((c2,outputs.conv2.cpu().detach().numpy()))
    c3=np.vstack((c3,outputs.conv3.cpu().detach().numpy()))
    c4=np.vstack((c4,outputs.conv4.cpu().detach().numpy()))
    c5=np.vstack((c5,outputs.conv5.cpu().detach().numpy()))


c1 = np.delete(c1,0,0); c2 = np.delete(c2,0,0); c3 = np.delete(c3,0,0); c4 = np.delete(c4,0,0); c5 = np.delete(c5,0,0);
c1 = c1[:,:,15:39,15:39]; c2 = c2[:,:,5:20, 5:20]; c3 = c3[:,:,3:11,3:11]; c4 = c4[:,:,3:11,3:11]; c5 = c5[:,:,3:11,3:11]; #grab just img center info
c1 = c1.reshape(551,-1); c2 = c2.reshape(551,-1); c3 = c3.reshape(551,-1); c4 = c4.reshape(551,-1); c5 = c5.reshape(551,-1); #reshape into n samples x nfeatures

conv_train = {'conv1': c1, 'conv2': c2, 'conv3': c3, 'conv4': c4, 'conv5': c5}


c1 = np.empty(shape=(1,64,55,55)); c2 = np.empty(shape=(1,192,27,27)); c3 = np.empty(shape=(1,384,13,13));
c4 = np.empty(shape=(1,256,13,13)); c5 = np.empty(shape=(1,256,13,13));



for i, data in enumerate(testloader):

    inputs = data['image']
    inputs=inputs.to(device)

    outputs = net(inputs)

    c1=np.vstack((c1,outputs.conv1.cpu().detach().numpy()))
    c2=np.vstack((c2,outputs.conv2.cpu().detach().numpy()))
    c3=np.vstack((c3,outputs.conv3.cpu().detach().numpy()))
    c4=np.vstack((c4,outputs.conv4.cpu().detach().numpy()))
    c5=np.vstack((c5,outputs.conv5.cpu().detach().numpy()))

c1 = np.delete(c1,0,0); c2 = np.delete(c2,0,0); c3 = np.delete(c3,0,0); c4 = np.delete(c4,0,0); c5 = np.delete(c5,0,0);
c1 = c1[:,:,15:39,15:39]; c2 = c2[:,:,5:20, 5:20]; c3 = c3[:,:,3:11,3:11]; c4 = c4[:,:,3:11,3:11]; c5 = c5[:,:,3:11,3:11]; #grab just img center info
c1 = c1.reshape(50,-1); c2 = c2.reshape(50,-1); c3 = c3.reshape(50,-1); c4 = c4.reshape(50,-1); c5 = c5.reshape(50,-1); #reshape into n samples x nfeatures

conv_test = {'conv1': c1, 'conv2': c2, 'conv3': c3, 'conv4': c4, 'conv5': c5}

np.save('data/conv_train.npy',conv_train)
np.save('data/conv_test.npy',conv_test)
"""
#%% Now fit the data


best_r2 = 0

results = dict()

for nnn in trange(1,df_now.shape[1],ncols=15):
    best_r2 = 0

    for modelnum in trange(len(mods),ncols=15):

        model = mods[modelnum]
        model_params = all_params[modelnum]

        y_full = df_now.iloc[:,nnn]

        good = ~np.isnan(y_full)

        ytrain = np.array(y_full.loc[good])

        for layer in tqdm(conv_train.keys(),ncols=15):
            xtrain = conv_train[layer][good,:].flatten().reshape(551,-1)
            xtest = conv_test[layer].flatten().reshape(50,-1)
            fun = train_models_fun(model,xtrain,ytrain)
            net_opt = BayesianOptimization(fun,model_params,verbose=0)
            net_opt.maximize(n_iter=50,acq="poi",xi=1e-1)

            r2_test = net_opt.max['target']
            best_params = net_opt.max['params']
            model.set_params(**best_params)

            if r2_test > best_r2:
                results[model+'_'+layer] = r2_test
                model.fit(xtrain,ytrain)
                out = model.predict(xtest)
                test.iloc[:,nnn] = out
                best_r2 = r2_test

test.to_csv('data/output.csv',index=False)
