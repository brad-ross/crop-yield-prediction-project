import os
import numpy as np
import sys

src = sys.argv[1]
ref = sys.argv[2]
dst = sys.argv[3]

def splice_data(src, ref, dst):
    ref_data = np.load(os.path.expanduser(ref))
    csv = np.genfromtxt(os.path.expanduser(src), delimiter=',')
    
    # Unpack the data from the .npz
    ref_index = ref_data['output_index']
    ref_year = ref_data['output_year']
    ref_image = ref_data['output_image']
    ref_locations = ref_data['output_locations']
    ref_yield = ref_data['output_yield']

    # Concatenate ref_year and ref_index for comparison
    temp = ref_year[:, np.newaxis]
    ref_search = np.concatenate([temp, ref_index], axis=1)
    indices = []
    csv_indices = []
    for i in range(csv.shape[0]):
        if i % 1000 == 0:
            print("On iteration %d." % i)
        result = np.where(np.all(ref_search == csv[i,:-1], axis=1))[0]
        if result.shape[0] == 0: # No match found
            print("ERROR, could not find %s" % csv[i])
            continue
        found_i = result[0]
        indices.append(found_i)
        csv_indices.append(i)

    # After this step, indices will have the indices from ref that we want to include.
    out_index = ref_index[indices]
    out_year = ref_year[indices]
    out_image = ref_image[indices]
    out_locations = ref_locations[indices]
    out_yield = csv[csv_indices, 3]
    assert(len(indices) == len(csv_indices))
    np.savez(os.path.expanduser(dst),
             output_index=out_index,
             output_year=out_year,
             output_image=out_image,
             output_locations=out_locations,
             output_yield=out_yield)

splice_data(src, ref, dst)
        
