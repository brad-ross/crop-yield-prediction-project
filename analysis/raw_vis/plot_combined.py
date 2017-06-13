import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import sys

loc1 = int(sys.argv[1])
loc2 = int(sys.argv[2])
year = int(sys.argv[3])
time = int(sys.argv[4])
band = int(sys.argv[5]) if len(sys.argv) > 5 else -1

raw = np.load('raw_%d_%d_%d.npz' % (year, loc1, loc2))
temp = np.load('raw_temp_%d_%d_%d.npz' % (year, loc1, loc2))
# vis = np.load('../vis/smap_vis_%d_%d_%d.npz' % (year, loc1, loc2))

def plot_rgb(time):
    r_slice = raw['raw_%d_t%d_b0' % (year, time)]
    g_slice = raw['raw_%d_t%d_b3' % (year, time)]
    b_slice = raw['raw_%d_t%d_b2' % (year, time)]
    img_array = np.stack([r_slice, g_slice, b_slice], axis=2)
    img_array = np.interp(img_array, [np.min(img_array), np.max(img_array)], [0, 255]).astype(np.uint8)
    plt.axis('off')
    plt.imshow(img_array)
    plt.show()
    plt.savefig('reflectance_%d_%d_%d.png' % (year, loc1, loc2), bbox_inches='tight', pad_inches=0)
    #img = Image.fromarray(img_array)
    #img.show()

def plot_temp(time):
    temp_slice = temp['raw_%d_t%d_b0' % (year, time)].astype(np.float64)
    mask = (temp_slice != 0)
    temp_slice_flat = temp_slice.reshape(-1)
    temp_slice_flat = temp_slice_flat[temp_slice_flat != 0]
    min_val = np.min(temp_slice_flat)
    max_val = np.max(temp_slice_flat)
    temp_slice -= min_val
    temp_slice *= (255 / (float(max_val) - min_val))
    temp_slice *= mask
    temp_slice = temp_slice.astype(np.uint8)
    # img = Image.fromarray(temp_slice)
    plt.axis('off')
    plt.imshow(temp_slice, vmin=0, vmax=255)
    plt.savefig('temperature_%d_%d_%d.png' % (year, loc1, loc2), bbox_inches='tight', pad_inches=0)
    # img.show(cmap='greens')

def make_overlay(time, band):
    r_slice = raw['raw_%d_t%d_b0' % (year, time)]
    g_slice = raw['raw_%d_t%d_b3' % (year, time)]
    b_slice = raw['raw_%d_t%d_b2' % (year, time)]
    img_array = np.stack([r_slice, g_slice, b_slice], axis=2)
    img_array = np.interp(img_array, [np.min(img_array), np.max(img_array)], [0, 255]).astype(np.uint8)
    print np.max(img_array)
    print np.min(img_array)
    vis = np.load('../vis/smap_vis_%d_%d_%d_ic.npz' % (year, loc1, loc2))
    soy_mask = (vis['vis_soy_t%d_b%d' % (time, band)] != 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(soy_mask.astype(np.uint8))
    vis_soy = np.interp(vis['vis_soy_t%d_b%d' % (time, band)], [-0.05, 0.05], [0, 255])
    vis_corn = np.interp(vis['vis_corn_t%d_b%d' % (time, band)], [-0.05, 0.05], [0, 255])
    soy_img = Image.fromarray(vis_soy).convert('RGBA')
    corn_img = Image.fromarray(vis_corn).convert('RGBA')
    img = Image.fromarray(img_array)
    img.paste(soy_img, (0, 0), mask_img)
    img = img.resize((400, 300), Image.ANTIALIAS)
    img.show()

#make_overlay(time, band)
plot_rgb(time)
plot_temp(time)   
# Code graveyard
'''
raw_img = Image.fromarray(raw_image_slice).convert("RGBA")
vis_img = Image.fromarray(vis_image_slice).convert("RGBA")

raw_img.paste(vis_img, (0, 0), vis_img)

raw_img.show()
vis_image_slice = vis['vis_soy_t31_b6']
'''
