import os
import numpy as np
import sys

# Gather stats about input data. Since there are way too many files to analyze,
# randomly select 1000 files.

outpath = os.path.expanduser(sys.argv[1])
MAX_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else 1000000

def gather_stats(pathname, outpath, MAX_FILES):
    num_files = 0
    IMAGE_PATH = os.path.expanduser(pathname)
    heights = []
    widths = []
    depths = []
    aspect_ratios = []
    prop_nonzero_pixels = []
    sizes = []
    with open(os.path.expanduser('~/crop-yield-prediction-project/analysis/clean_filenames')) as f:
        filenames = f.readlines()
    if len(filenames) > MAX_FILES:
        filename = list(np.random.choice(filenames, MAX_FILES, replace=False))
    for filename in filenames:
        if num_files % 100 == 0:
            print "Iteration %d." % num_files 
        if num_files == MAX_FILES:
            break
        filename = filename.split('\n')[0]
        if filename.endswith('.npy'):
            num_files += 1
            arr = np.load(os.path.join(IMAGE_PATH, filename))
            heights.append(arr.shape[0])
            widths.append(arr.shape[1])
            aspect_ratios.append(arr.shape[1] / float(arr.shape[0]))
            num_nonzero_pixels = np.sum(arr > 0)
            prop_nonzero_pixels.append(num_nonzero_pixels / (float(arr.shape[0])* arr.shape[1]*arr.shape[2]))
            depths.append(arr.shape[2])
            sizes.append(os.path.getsize(os.path.join(IMAGE_PATH, filename)))
        np.savez(outpath,
                 heights=np.array(heights),
                 widths=np.array(widths),
                 depths=np.array(depths),
                 aspect_ratios=np.array(aspect_ratios),
                 prop_nonzero_pixels=np.array(prop_nonzero_pixels),
                 sizes=np.array(sizes))
            

gather_stats('~/cs231n-satellite-images-clean', outpath, MAX_FILES)
