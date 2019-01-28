#!/usr/bin/env python3

import numpy as np
import pywt
from scipy.stats import kurtosis, skew

def raw_pixels(image):
    return image.flatten()

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

def mean_pixels(image): #mean pixel value of image
    return [np.mean(image)]

def std_pixels(image): #std of pixel values
    return [np.std(image)]

def kurt(image):
    return [kurtosis(image,axis=None)]

def skw(image):
    return [skew(image,axis=None)]

def min(image):
    im = image[20:60,20:60,:]
    return [np.min(im,axis=None)]

def max(image):
    im = image[20:60,20:60,:]
    return [np.min(image,axis=None)]

def rms(image): #root mean square
    return [np.sqrt(np.mean(image**2))]

