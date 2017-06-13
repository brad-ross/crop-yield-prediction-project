import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import gdal

loc1 = int(sys.argv[1])
loc2 = int(sys.argv[2])
year = int(sys.argv[3])

RAW_IMG_PATH = os.path.expanduser('~/cs231n-satellite-images/data_image_full_%d_%d.tif' % (loc1, loc2))
raw_arr = np.array(gdal.Open(RAW_IMG_PATH).ReadAsArray(), dtype='uint16')
raw_arr_year = raw_arr[46*7*(year-2003):46*7*(year-2003+1)] # Depth 414
raw_arr_year = raw_arr_year[42:266] # As per their magic numbers

NUM_TIMES = 32
NUM_BANDS = 7
entries = {}
entries['loc'] = np.array([loc1, loc2])
for t in range(NUM_TIMES):
    if t % 10 == 0:
        print 'On time %d.' % t
    for b in range(NUM_BANDS):
        raw_arr_slice = raw_arr_year[7*t+b]
        entries['raw_%d_t%d_b%d' % (year, t, b)] = raw_arr_slice
np.savez('raw_%d_%d_%d.npz' % (year, loc1, loc2), **entries)

RAW_TEMP_PATH = os.path.expanduser('~/cs231n-satellite-images/data_temperature_%d_%d.tif' % (loc1, loc2))
raw_temp_arr = np.array(gdal.Open(RAW_TEMP_PATH).ReadAsArray(), dtype='uint16')
raw_temp_arr_year = raw_temp_arr[46*2*(year-2003):46*2*(year-2003+1)] # Depth 414
raw_temp_arr_year = raw_temp_arr_year[12:76] # As per their magic numbers
NUM_TIMES = 32
NUM_BANDS = 2
entries = {}
entries['loc'] = np.array([loc1, loc2])
for t in range(NUM_TIMES):
    if t % 10 == 0:
        print 'On time %d.' % t
    for b in range(NUM_BANDS):
        raw_temp_arr_slice = raw_temp_arr_year[2*t+b]
        entries['raw_%d_t%d_b%d' % (year, t, b)] = raw_temp_arr_slice
np.savez('raw_temp_%d_%d_%d.npz' % (year, loc1, loc2), **entries)
