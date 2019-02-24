#!/usr/bin/env python3

from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, ElasticNetCV, LassoCV
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA

import umap

import pandas as pd
import numpy as np
from scipy import stats, signal

print('loading data...')
images = np.load('data/stim.npy')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/sub.csv')

dd = np.load('data/features.npz')
features = dict()
for k in dd.keys():
    features[k] = dd[k]

responses = np.array(train.iloc[:, 1:])


images_flat = images[:, 20:-20,20:-20].reshape(images.shape[0], -1)

print('computing UMAP...')
embedding = umap.UMAP(metric='correlation', min_dist=1.0, n_components=50)
X_embed = embedding.fit_transform(images_flat)

print('computing PCA...')
pcs_lab = PCA(n_components=50).fit_transform(features['LAB'])
# pcs_raw = PCA(n_components=50).fit_transform(features['raw'])
# pcs_fourier = PCA(n_components=10).fit_transform(features['fourier'])
pcs_gabor = PCA(n_components=20).fit_transform(features['gabor'])

X_all = np.hstack([X_embed, features['stats'], pcs_lab])
# X_all = np.hstack([X_embed, features['stats'], pcs_lab, pcs_fourier])

print('estimating errors...')

# model = Ridge(alpha=5)
# model = LassoCV()
# model = RidgeCV()
model = ExtraTreesRegressor(max_depth=15, n_estimators=100, n_jobs=16)
# model = KNeighborsRegressor(n_neighbors=100)
# model = SVR('rbf')

all_scores = []
for i in range(responses.shape[1]):
    vals = responses[:, i]
    good = ~np.isnan(vals)
    scores = cross_val_score(model, X_all[50:][good], vals[good],
                             scoring=make_scorer(mean_squared_error),
                             cv=ShuffleSplit(n_splits=3, test_size=0.1))
    print(i, np.mean(scores))
    all_scores.append(np.mean(scores))

print(np.mean(all_scores))
print(np.sqrt(np.mean(all_scores)))

print('computing and saving predictions...')
new_test = test.copy()

for i in range(responses.shape[1]):
    vals = responses[:, i]
    good = ~np.isnan(vals)
    model.fit(X_all[50:][good], vals[good])
    pred = model.predict(X_all[:50])
    new_test.iloc[:, i+1] = pred

new_test.to_csv('data/output_umap.csv', index=False)
