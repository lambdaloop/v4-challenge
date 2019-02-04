#!/usr/bin/env python3

import numpy as np
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics.scorer import make_scorer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.decomposition import PCA
import pandas as pd
from bayes_opt import BayesianOptimization
from tqdm import tqdm, trange

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/sub.csv')

dd = np.load('data/features.npz')
features = dict()
for k in dd.keys():
    features[k] = dd[k]

models = [Lasso(alpha=1000000), Ridge(alpha=100000), ElasticNet(),
          RandomForestRegressor(max_depth=7), ExtraTreesRegressor()]

all_params = [{'alpha': (0.01, 1e6)}, {'alpha': (0.01, 1e6)}, {'alpha': (0.01, 1e6)},
              {'max_depth': (3, 15)}, {'max_depth': (3, 15)}]


print(features.keys())
# ['raw', 'LAB', 'fourier', 'gabor', 'stats']

def train_models_fun(model, X_full, y_full):
    def test_model(**params):
        model.set_params(**params)
        scores = cross_val_score(model, X_full, y_full,
                                 cv=ShuffleSplit(n_splits=1, test_size=0.1, random_state=42),
                                 scoring=make_scorer(r2_score))
        r2_test = np.mean(scores)
        return r2_test
    return test_model


for neuron_number in trange(1, train.shape[1], ncols=80):
    # print('neuron number:', neuron_number)
    best_r2 = 0

    for modelnum in trange(len(models), ncols=80):
        # print('model num:', modelnum)
        model = models[modelnum]
        model_params = all_params[modelnum]

        for feature in tqdm(features.keys(), ncols=80):

            ids_train = np.array(train['Id'])
            ids_test = np.array(test['Id'])

            X_test = features[feature][ids_test]
            X_full = features[feature][ids_train]
            # X_full = np.hstack([features['stats'][ids_train],
            #                     features[feature][ids_train]])
            y_full = np.array(train.iloc[:, neuron_number])

            good = ~np.isnan(y_full)
            X_full = X_full[good]
            y_full = y_full[good]

            good = np.std(X_full, axis=0) > 0
            X_full = X_full[:, good]
            X_test = X_test[:, good]

            X_all = np.vstack([X_full, X_test])
            pca = PCA(n_components=min(X_full.shape[1], 100))
            pca.fit(X_all)
            X_full = pca.transform(X_full)
            X_test = pca.transform(X_test)

            fun = train_models_fun(model, X_full, y_full)
            net_opt = BayesianOptimization(fun,model_params,verbose=False)
            net_opt.maximize(init_points=5,n_iter=25,acq="ei",xi=1e-4)

            r2_test = net_opt.max['target']
            best_params = net_opt.max['params']
            model.set_params(**best_params)
            # print('feature: {}, r2: {:.3f}'.format(feature, r2_test))

            if r2_test > best_r2:
                model.fit(X_full, y_full)
                out = model.predict(X_test)
                test.iloc[:, neuron_number] = out
                best_r2 = r2_test

test.to_csv('data/output.csv', index=False)
