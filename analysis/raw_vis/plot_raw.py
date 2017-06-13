import numpy as np
import matplotlib.pyplot as plt
import os

raw = np.load('raw_2013_1_1.npz')
image_slice = raw['raw_2013_t31_b6']
plt.imshow(image_slice, cmap='gray')
plt.show()
