#!/usr/bin/env python3

import numpy as np
import pywt

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

