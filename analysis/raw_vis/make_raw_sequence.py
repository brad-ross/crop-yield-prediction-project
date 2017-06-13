import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import sys

loc1 = int(sys.argv[1])
loc2 = int(sys.argv[2])
year = 2013

raw = np.load('raw_%d_%d_%d.npz' % (year, loc1, loc2))
temp = np.load('raw_temp_%d_%d_%d.npz' % (year, loc1, loc2))

dirname = 'raw_seq_%d_%d' % (loc1, loc2)
if not os.path.exists(dirname):
    os.makedirs(dirname)

def plot_rgb(time):
    r_slice = raw['raw_%d_t%d_b0' % (year, time)]
    g_slice = raw['raw_%d_t%d_b3' % (year, time)]
    b_slice = raw['raw_%d_t%d_b2' % (year, time)]
    img_array = np.stack([r_slice, g_slice, b_slice], axis=2)
    img_array = np.interp(img_array, [np.min(img_array), np.max(img_array)], [0, 255]).astype(np.uint8)
    plt.axis('off')
    plt.title('Raw (t=%d)' % time)
    plt.imshow(img_array, vmin=0, vmax=255)
    # plt.show()
    plt.savefig(os.path.join(dirname, 't%d.png' % time), bbox_inches='tight', pad_inches=0)

NUM_TIMES = 32
for t in range(NUM_TIMES):
    plot_rgb(t)
