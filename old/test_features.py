#!/usr/bin/env python3

## Pierre's code to test feature stuff

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pywt
from sklearn import decomposition

images = np.load('data/stim.npy')

img = images[100]

fft = np.fft.fft2(img)
fft_mag = np.abs(fft)
fft_ang = np.angle(fft)

cA, (cH, cV, cD) = pywt.dwt2(img[:, :, 0], 'sym4')

plt.figure(1)
plt.clf()
plt.imshow(img[10:-10,10:-10])
plt.draw()
plt.show(block=False)

gray= cv.cvtColor(img,cv.COLOR_RGB2GRAY)
gray8 = np.uint8(gray*255)
sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray8,None)

pca = decomposition.PCA()
img_transform = pca.fit_transform(images.reshape(images.shape[0], -1))

comps = pca.components_.reshape((601,) + img.shape)

def normalize(X, p=1):
    a, b = np.percentile(X, [p,100-p])
    return (X - a) / (b - a)

plt.figure(2)
plt.clf()
for i in range(16):
    plt.subplot(4,4,i+1)
    # c = normalize(comps[i],p=2)*255
    c = comps[i+64]*100
    plt.imshow(c)
    plt.axis('off')
plt.draw()
plt.tight_layout()
plt.show(block=False)

for i in range(comps.shape[0]):
    c = comps[i]*100
    img2 = cv.cvtColor(c, cv.COLOR_RGB2BGR)
    imgx = np.clip(img2 * 255, 0, 255)

    fname = 'data/components/comp{:03d}.png'.format(i)
    cv.imwrite(fname, imgx)
