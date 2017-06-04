import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

year = 2013
loc1 = 1
loc2 = 1

raw = np.load('raw_%d_%d_%d.npz' % (year, loc1, loc2))
vis = np.load('../vis/smap_vis_%d_%d_%d.npz' % (year, loc1, loc2))

def plot_rgb(time):
    r_slice = raw['raw_%d_t%d_b0' % (year, time)]
    g_slice = raw['raw_%d_t%d_b3' % (year, time)]
    b_slice = raw['raw_%d_t%d_b2' % (year, time)]
    img_array = np.stack([r_slice, g_slice, b_slice], axis=2)
    img_array = np.interp(img_array, [np.min(img_array), np.max(img_array)], [0, 255]).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.show()

plot_rgb(31)
    
# Code graveyard
'''
raw_img = Image.fromarray(raw_image_slice).convert("RGBA")
vis_img = Image.fromarray(vis_image_slice).convert("RGBA")

raw_img.paste(vis_img, (0, 0), vis_img)

raw_img.show()
vis_image_slice = vis['vis_soy_t31_b6']
'''
