import numpy as np
import matplotlib.pyplot as plt
import sys
import os

FILE_PATH = os.path.expanduser(sys.argv[1])
vis = np.load(FILE_PATH)
f, axarr = plt.subplots(1, 2)
axarr[0].imshow(vis['vis_corn_n'], cmap='gray')
axarr[1].imshow(vis['vis_soy_n'], cmap='gray')
plt.show()
