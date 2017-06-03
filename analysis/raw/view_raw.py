import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import gdal

loc1 = int(sys.argv[1])
loc2 = int(sys.argv[2])
year = int(sys.argv[3])
time = int(sys.argv[4]) # MAX 32
band = int(sys.argv[5]) # HAS TO BE 0-6 inclusive!

RAW_PATH = os.path.expanduser('~/cs231n-satellite-images/data_image_full_%d_%d.tif' % (loc1, loc2))
raw_arr = np.array(gdal.Open(RAW_PATH).ReadAsArray(), dtype='uint16')
raw_arr_year = raw_arr[46*7*(year-2003):46*7*(year-2003+1)] # Depth 414
raw_arr_year = raw_arr_year[54:342] # As per their magic numbers

NUM_TIMES = 32
NUM_BANDS = 7
entries = {}
entries['loc'] = np.array([loc1, loc2])
for t in range(NUM_TIMES):
    if t % 10 == 0:
        print 'On time %d.' % t
    for b in range(NUM_BANDS):
        raw_arr_slice = raw_arr_year[7*t+b]
        entries['raw_%d_t%d_b%d' % (year, time, band)] = raw_arr_slice
np.savez('raw_%d_%d_%d.npz' % (year, loc1, loc2), **entries)
