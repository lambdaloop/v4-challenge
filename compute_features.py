#!/usr/bin/env python3

import numpy as np
import pywt
from scipy.stats import kurtosis, skew
import cv2 as cv

def raw_pixels(image):
    return image.flatten()

def colorspace_image(image, colorspace='LAB'):
    if colorspace == 'LAB':
        return cv.cvtColor(image, cv.COLOR_RGB2LAB)
    elif colorspace == 'HSV':
        return cv.cvtColor(image, cv.COLOR_RGB2HSV)
    else:
        return image

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


FEATURE_FUNCTIONS = {
    'raw': raw_pixels,
    'LAB': (colorspace_image, 'LAB'),
    'fourier': fourier_features,
    'gabor': (wavelet_features, 'sym4'),
    'stats': calc_stats
}

def get_features_image(image):
    out = dict()
    for label, ffun in FEATURE_FUNCTIONS.items():
        if isinstance(ffun, tuple):
            feature_fun =  ffun[0]
            args = ffun[1:]
        else:
            feature_fun = ffun
            args = ()
        out[label] =  feature_fun(img, *args)
    return out

def compute_features():
    images = np.load('data/stim.npy')
    out = []
    for i in range(images.shape[0]):
        img = images[i]
        features = get_features_image(img)
        out.append(features)
    return out

if __name__ == '__main__':
    features = compute_features()
    np.savez_compressed('data/features.npz', features)
