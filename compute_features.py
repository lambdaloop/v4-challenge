#!/usr/bin/env python3

import numpy as np
from scipy.stats import kurtosis, skew

def raw_pixels(image):
    return image

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
