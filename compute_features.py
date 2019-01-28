#!/usr/bin/env python3

import numpy as np
from scipy.stats import kurtosis, skew

def raw_pixels(image):
    return image

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
