#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

images = np.load('data/stim.npy')

plt.figure(1)
plt.clf()
plt.imshow(images[5])
plt.draw()
plt.show(block=False)

