import os
import numpy as np
import sys
import back_out as bk

if len(sys.argv) == 1:
    print "Usage: python create_smap_vis.py -a [loc1] [loc2]"
    print "       python create_smap_vis.py -s [time] [band] [num]"
    print "TLDR: -a iterates over time/band given location."
    print "      -s iterates over num pairs of loc1/loc2 given time/band."
    exit()
    
assert(sys.argv[1] == '-a' or sys.argv[1] == '-s')
    
MAP_PATH = os.path.expanduser('~/cs231n-satellite-images-models/saliency_maps/original_model_comparison_imp_count.npz')
REF_PATH = os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean_filtered.npz')

maps = np.load(MAP_PATH)
ref = np.load(REF_PATH)
#indices = ref['output_index'][ref['output_year'] == 2013]
#important_counties = [5, 17, 18, 19, 20, 27, 29, 31, 38, 39, 46]
#imp_indices = []
#for row in indices:
#    if row[0] in important_counties:
#        imp_indices.append(row)
#indices = np.stack(imp_indices, axis=0)
indices = np.load('../unbroken_index.npy')
corn_maps = maps['corn_maps']
soy_maps = maps['soy_maps']
print corn_maps.shape
print indices.shape
assert(corn_maps.shape[0] == indices.shape[0])
assert(soy_maps.shape[0] == indices.shape[0])
global_min = min(np.min(corn_maps), np.min(soy_maps))
global_max = max(np.max(corn_maps), np.max(soy_maps))
val_range = [global_min, global_max]

def create_single_vis(loc1, loc2, time, band): # loc1 is state, loc2 is county
    global corn_maps
    global soy_maps
    global indices
    global val_range

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
    
    # Normalize the gradients based on the maximum and minimum values in ALL maps
    vis_corn_n = bk.freq_to_intens(val_range, vis_corn).astype(np.uint8)
    vis_soy_n = bk.freq_to_intens(val_range, vis_soy).astype(np.uint8)
    
    return (vis_corn, vis_corn_n, vis_soy, vis_soy_n, image_slice)

def create_all_vis(loc1, loc2):
    NUM_TIMES = 32
    NUM_BANDS = 9
    vis_thingy = []
    for i in range(NUM_TIMES * NUM_BANDS):
        if i % 10 == 0:
            print 'Creating vis %d of %d.' % (i, NUM_TIMES * NUM_BANDS)
        vis_thingy.append(create_single_vis(loc1, loc2, i / 9, i % 9))
    return vis_thingy

def package_all_vis(loc1, loc2, vis_thingy): # Need to pass in loc1, loc2 to label it properly
    entries = {}
    entries['loc'] = np.array([loc1, loc2])
    assert(len(vis_thingy) == 288)
    for i, vis_tuple in enumerate(vis_thingy):
        vis_corn, vis_corn_n, vis_soy, vis_soy_n, image_slice = vis_tuple
        entries['vis_corn_t%d_b%d' % (i / 9, i % 9)] = vis_corn
        entries['vis_corn_n_t%d_b%d' % (i / 9, i % 9)] = vis_corn_n
        entries['vis_soy_t%d_b%d' % (i / 9, i % 9)] = vis_soy
        entries['vis_soy_n_t%d_b%d' % (i / 9, i % 9)] = vis_soy_n
        entries['raw_t%d_b%d' % (i / 9, i % 9)] = image_slice
    np.savez('smap_vis_2013_%d_%d_ic.npz' % (loc1, loc2), **entries)

def create_sample_vis(time, band, num=100):
    global indices
    assert(num <= indices.shape[0])
    np.random.seed(231)
    selected_indices = indices[np.random.choice(indices.shape[0], num, replace=False)]
    vis_thingy2 = []
    for i in range(selected_indices.shape[0]):
        if i % 10 == 0:
            print 'Creating vis %d of %d.' % (i, selected_indices.shape[0])
        vis_thingy2.append(create_single_vis(selected_indices[i,0], selected_indices[i,1], time, band))
    return (selected_indices, vis_thingy2)

def package_sample_vis(time, band, selected_indices, vis_thingy2):
    entries = {}
    entries['indices'] = selected_indices
    assert(len(vis_thingy2) == selected_indices.shape[0])
    for i, vis_tuple in enumerate(vis_thingy2):
        vis_corn, vis_corn_n, vis_soy, vis_soy_n, image_slice = vis_tuple
        entries['vis_corn_%d_%d' % (selected_indices[i,0], selected_indices[i,1])] = vis_corn
        entries['vis_corn_n_%d_%d' % (selected_indices[i,0], selected_indices[i,1])] = vis_corn_n
        entries['vis_soy_%d_%d' % (selected_indices[i,0], selected_indices[i,1])] = vis_soy
        entries['vis_soy_n_%d_%d' % (selected_indices[i,0], selected_indices[i,1])] = vis_soy_n
        entries['raw_%d_%d' % (selected_indices[i,0], selected_indices[i,1])] = image_slice
    np.savez('smap_vis_2013_t%d_b%d_ic.npz' % (time, band), **entries)

if sys.argv[1] == '-a':
    loc1 = int(sys.argv[2])
    loc2 = int(sys.argv[3])
    print "Creating visualizations..."
    vis_thingy = create_all_vis(loc1, loc2)
    print "Packaging visualizations..."
    package_all_vis(loc1, loc2, vis_thingy)
    print "Done."
elif sys.argv[1] == '-s':
    time = int(sys.argv[2])
    band = int(sys.argv[3])
    num = int(sys.argv[4])
    print "Creating visualizations..."
    vis_blob = create_sample_vis(time, band, num)
    print "Packaging visualizations..."
    package_sample_vis(time, band, *vis_blob)
    print "Done."
