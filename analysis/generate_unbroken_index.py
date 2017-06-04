import numpy as np
import os
import pandas as pd

VAL_YEAR = 2013
states_to_keep = np.array([5, 17, 18, 19, 20, 27, 29, 31, 38, 39, 46])

# Soybean Saliency Maps
soy_data = np.load(os.path.expanduser('~/cs231n-satellite-images-hist/data_soybean_filtered.npz'))

hist_sums = np.sum(soy_data['output_image'],axis=(1,2,3))
nonbroken_rows = hist_sums > 287
imp_rows = pd.DataFrame(soy_data['output_index'])[0].isin(states_to_keep)
val_year_rows = soy_data['output_year'] == VAL_YEAR
index_validate = np.logical_and.reduce((nonbroken_rows, imp_rows, val_year_rows))

unbroken_index = soy_data['output_index'][index_validate]
print unbroken_index.shape

np.save('unbroken_index.npy', unbroken_index)
