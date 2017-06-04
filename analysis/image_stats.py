import os
import numpy as np
import sys
import re
import gdal

# Gather stats about input data. Since there are way too many files to analyze,
# randomly select 1000 files.

outpath = os.path.expanduser(sys.argv[1])
MAX_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000

def gather_stats(pathname, raw_pathname, outpath, MAX_FILES):
    num_files = 0
    IMAGE_PATH = os.path.expanduser(pathname)
    RAW_PATH = os.path.expanduser(raw_pathname)
    heights = []
    widths = []
    depths = []
    aspect_ratios = []
    prop_nonzero_pixels = []
    sizes = []
    with open(os.path.expanduser('~/crop-yield-prediction-project/analysis/clean_filenames')) as f:
        filenames_all = f.readlines()
    filenames = []
    for filename in filenames_all:
        if 'zoom' not in filename:
            filenames.append(filename)
    if len(filenames) > MAX_FILES:
        filename = list(np.random.choice(filenames, MAX_FILES, replace=False))
    for filename in filenames:
        if num_files % 10 == 0:
            print "Iteration %d." % num_files 
        if num_files == MAX_FILES:
            break
        filename = filename.split('\n')[0]
        if filename.endswith('.npy'):
            num_files += 1
            arr = np.load(os.path.join(IMAGE_PATH, filename))
            arr = arr[:,:,54:342]
            raw_filename = 'data_image_full_' + filename[22:-4] + '.tif'
            # print os.path.join(RAW_PATH, raw_filename)
            raw_arr = np.transpose(np.array(gdal.Open(os.path.join(RAW_PATH, raw_filename)).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
            year = int(filename[17:21])
            raw_arr = raw_arr[:,:,322*(year-2003):322*(year-2003+1)]
            num_nonzero_pixels_raw = np.sum(raw_arr[:,:,0] > 0)
            heights.append(arr.shape[0])
            widths.append(arr.shape[1])
            aspect_ratios.append(arr.shape[1] / float(arr.shape[0]))
            num_nonzero_pixels = np.sum(arr > 0) / arr.shape[2]
            prop_nonzero_pixels.append(num_nonzero_pixels / float(num_nonzero_pixels_raw))
            depths.append(arr.shape[2])
            sizes.append(os.path.getsize(os.path.join(IMAGE_PATH, filename)))
        np.savez(outpath,
                 heights=np.array(heights),
                 widths=np.array(widths),
                 depths=np.array(depths),
                 aspect_ratios=np.array(aspect_ratios),
                 prop_nonzero_pixels=np.array(prop_nonzero_pixels),
                 sizes=np.array(sizes))
            

gather_stats('~/cs231n-satellite-images-clean', '~/cs231n-satellite-images', outpath, MAX_FILES)
