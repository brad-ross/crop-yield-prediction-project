import os
import numpy as np
import sys

    
MAP_PATH = os.path.expanduser('~/cs231n-satellite-images-models/saliency_maps/original_model_comparison_imp_count.npz')
REF_PATH = os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean_filtered.npz')

maps = np.load(MAP_PATH)
ref = np.load(REF_PATH)

indices = np.load('../unbroken_index.npy')
corn_maps = maps['corn_maps']
soy_maps = maps['soy_maps']

def compute_stats():
    diffs = np.sqrt(np.mean((corn_maps - soy_maps) ** 2, axis=1)) 
    print "The maximum difference is %f, and occurs at:" % np.max(diffs)
    loc_max = np.unravel_index(np.argmax(diffs), diffs.shape)
    ind_max = indices[loc_max[0]]
    print (int(ind_max[0]), int(ind_max[1]), loc_max[1], loc_max[2])
    print "The minimum difference is %f, and occurs at:" % np.min(diffs)
    loc_min = np.unravel_index(np.argmin(diffs), diffs.shape)
    ind_min = indices[loc_min[0]]
    print (int(ind_min[0]), int(ind_min[1]), loc_min[1], loc_min[2])   

compute_stats()
