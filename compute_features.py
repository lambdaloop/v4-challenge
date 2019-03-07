#!/usr/bin/env python3

import numpy as np
import pywt
from scipy.stats import kurtosis, skew, moment
import cv2 as cv
from tqdm import trange
from collections import defaultdict
from bayes_opt import BayesianOptimization

def raw_pixels(image):
    return image.flatten()

def get_edges(im):
    dx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=3)
    dy = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=3)
    absr = np.abs(dx + dy*1j)
    return absr

def colorspace_image(image, colorspace='LAB'):
    if colorspace == 'LAB':
        out = cv.cvtColor(image, cv.COLOR_RGB2LAB)
    elif colorspace == 'HSV':
        out = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    elif colorspace == 'edges':
        image = get_edges(img)
    else:
        out = image

    return out.flatten()

def fourier_features(img):
    fft = np.fft.fft2(img)
    fft_mag = np.abs(fft)
    # fft_ang = np.angle(fft)
    return fft_mag.flatten()

def wavelet_features(img, wavelet='sym4'):
    features = []
    for i in range(3):
        cA, (cH, cV, cD) = pywt.dwt2(img[:, :, i], 'sym4')
        features.extend([cA, cH, cV, cD])
    return np.hstack(features).flatten()
    

def calc_stats(img, colorspace=None):
    if colorspace == 'LAB':
        image = cv.cvtColor(img, cv.COLOR_RGB2LAB)
    elif colorspace == 'HSV':
        image = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    elif colorspace == 'edges':
        image = get_edges(img)
    else:
        image = img

    results = []

    for j in range(3):
        im = image[:, :, j]
        results.append(np.min(im))
        results.append(np.max(im))
        results.append(np.sqrt(np.mean(np.square(im))))

        results.append(np.mean(im))
        results.append(np.std(im))
        if np.abs(np.std(im)) < 1e-12:
            im_std = im - np.mean(im)
        else:
            im_std = (im - np.mean(im)) / np.std(im)
        for mnum in range(3, 8):
            results.append(moment(im_std, moment=mnum, axis=None))

    return np.array(results)


FEATURE_FUNCTIONS = {
    'raw': raw_pixels,
    'LAB': (colorspace_image, 'LAB'),
    'fourier': fourier_features,
    'gabor': (wavelet_features, 'sym4'),
    'stats': calc_stats,
    'stats_LAB': (calc_stats, 'LAB'),
    'stats_HSV': (calc_stats, 'HSV'),
    'stats_edges': (calc_stats, 'edges')
}

def get_features_image(img):
    out = dict()
    for label, ffun in FEATURE_FUNCTIONS.items():
        if isinstance(ffun, tuple):
            feature_fun = ffun[0]
            args = ffun[1:]
        else:
            feature_fun = ffun
            args = ()
        out[label] = feature_fun(img, *args)
    return out

def compute_features():
    images = np.load('data/stim.npy')
    out = defaultdict(list)
    for i in trange(images.shape[0]):
        img = images[i]
        features = get_features_image(img[20:-20,20:-20])
        for k, v in features.items():
            out[k].append(v)
    for k,v in out.items():
        out[k] = np.array(v)
    return out

if __name__ == '__main__':
    features = compute_features()
    np.savez_compressed('data/features.npz', **features)
