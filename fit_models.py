#!/usr/bin/env python3

import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import pandas as pd

train = pd.read_csv('data/train.csv')

dd = np.load('data/features.npz')
features = dict()
for k in dd.keys():
    features[k] = dd[k]

models = [Lasso(alpha=10), Ridge(alpha=1000), ElasticNet(), RandomForestRegressor(max_depth=5), ExtraTreesRegressor()]

model = models[1]
feature = 'stats'

ids = np.array(train['Id'])

X_full = features[feature][ids]
y_full = np.array(train.iloc[:, 1])

good = np.std(X_full, axis=0) > 0
X_full = X_full[:, good]

scores = cross_val_score(model, X_full, y_full, cv=3, scoring=make_scorer(r2_score))

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(r2)
