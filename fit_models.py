#!/usr/bin/env python3

import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA
import pandas as pd


train = pd.read_csv('data/train.csv')

# dd = np.load('data/features.npz')
# features = dict()
# for k in dd.keys():
#     features[k] = dd[k]

models = [Lasso(alpha=1000000), Ridge(alpha=100), ElasticNet(), RandomForestRegressor(max_depth=7), ExtraTreesRegressor()]

# things to iterate over
model = models[1]
feature = 'gabor'
neuron_number = 0

ids = np.array(train['Id'])

# X_full = features[feature][ids]
X_full = np.hstack([features['stats'][ids], features[feature][ids]])
y_full = np.array(train.iloc[:, neuron_number])

good = ~np.isnan(y_full)
X_full = X_full[good]
y_full = y_full[good]

good = np.std(X_full, axis=0) > 0
X_full = X_full[:, good]

pca = PCA(n_components=min(X_full.shape[1], 10))
X_full = pca.fit_transform(X_full)

scores = cross_val_score(model, X_full, y_full,
                         cv=ShuffleSplit(n_splits=3, test_size=0.1, random_state=42),
                         scoring=make_scorer(r2_score))
print(np.mean(scores))

# X_train, X_test, y_train, y_test = train_test_split(
#     X_full, y_full, test_size=0.1)

# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# r2 = r2_score(y_test, y_pred)

# print(r2)
