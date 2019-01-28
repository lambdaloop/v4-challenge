#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

images = np.load('data/stim.npy')

img = images[100]

plt.figure(1)
plt.clf()
plt.imshow(img)
plt.draw()
plt.show(block=False)

