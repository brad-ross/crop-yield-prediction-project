import os
import numpy as np
import sys
import back_out as bk

MAP_PATH = os.path.expanduser('~/cs231n-satellite-images-models/saliency_maps/original_model_comparison.npz')
REF_PATH = os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean_filtered.npz')

maps = np.load(MAP_PATH)
ref = np.load(REF_PATH)
indices = ref['output_index'][ref['output_year'] == 2013]

def visualize_maps(loc1, loc2, time, band): # loc1 is state, loc2 is county
    global maps
    global indices
    corn_maps = maps['corn_maps']
    soy_maps = maps['soy_maps']

    # Read in the cleaned image
    IMAGE_PATH = os.path.expanduser('~/cs231n-satellite-images-clean/data_output_full_%d_%d_%d.npy' % (2013, loc1, loc2))
    image = np.load(IMAGE_PATH)
    image = image[:,:,54:342] # Based on start_day = 49, end_day = 305
    assert(image.shape[2] == 32 * 9)
    
    # Select the correct histogram gradient
    i = np.where(np.all(indices == np.array([loc1, loc2]), axis=1))[0]
    assert(len(i) != 0)
    i = i[0]
    dhist_corn = corn_maps[i,:,time,band]
    dhist_soy = soy_maps[i,:,time,band]

    # Select the correct slice of the image and back out the gradients
    image_slice = image[:,:,9*time+band]
    vis_corn = bk.back_out_single(image_slice, dhist_corn)
    vis_soy = bk.back_out_single(image_slice, dhist_soy)
    
    # Normalize the gradients based on the maximum and minimum values in BOTH maps
    min_val = min(np.min(vis_corn), np.min(vis_soy))
    max_val = max(np.max(vis_corn), np.max(vis_soy))
    val_range = [min_val, max_val]
    vis_corn_n = bk.freq_to_intens(val_range, vis_corn).astype(np.uint8)
    vis_soy_n = bk.freq_to_intens(val_range, vis_soy).astype(np.uint8)
    np.savez('smap_vis_2013_%d_%d_t%d_b%d.npz' % (loc1, loc2, time, band),
             vis_corn=vis_corn,
             vis_corn_n=vis_corn_n,
             vis_soy=vis_soy,
             vis_soy_n=vis_soy_n)
    
visualize_maps(1, 1, 16, 5)
