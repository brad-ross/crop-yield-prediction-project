import pandas
import numpy as np
import math
import sys

if len(sys.argv) <= 3:
    print("Usage: python generate_csv.py [input] [reference] [output]")
    exit()
    
src = sys.argv[1]
ref = sys.argv[2]
dst = sys.argv[3]
print dst

src_data = np.genfromtxt(src, delimiter=',')
ref_data = np.genfromtxt(ref, delimiter=',')

def remove_nans(src_data):
    indices = []
    for i in range(src_data.shape[0]):
        if(math.isnan(src_data[i,2])):
	    continue
	indices.append(i)
    return src_data[indices]

def remove_years(src_data):
    years_to_remove = [2014, 2015, 2016]
    indices = []
    for i in range(src_data.shape[0]):
        if(src_data[i,0] in years_to_remove):
	    continue
	indices.append(i)
    return src_data[indices]

def filter_entries(src_data, ref_data):
    keys = {}
    for i in range(ref_data.shape[0]):
        key = (ref_data[i, 0], ref_data[i, 1], ref_data[i, 2])
        keys[key] = False
    indices = []
    for i in range(src_data.shape[0]):
        key = (src_data[i, 0], src_data[i, 1], src_data[i, 2])
        if key in keys:
            if not keys[key]:
                indices.append(i)
                keys[key] = True
            else: # DUPLICATED!
                print "DUPLICATION FOR %s!" % key
    return src_data[indices]

src_data = remove_nans(src_data)
src_data = remove_years(src_data)
ref_data = remove_nans(ref_data)
ref_data = remove_years(ref_data)
out_data = filter_entries(src_data, ref_data)
np.savetxt(dst, out_data, fmt=['%d','%d','%d','%.1f'], delimiter=',')
