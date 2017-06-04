import numpy as np
import matplotlib.pyplot as plt

output_rescaling_data = np.load('./poster/output_rescaling/original_model_rescaled_outputs.npz')
locs = output_rescaling_data['locs']
orig_corn_yield, orig_soy_yield = output_rescaling_data['corn_yield'], output_rescaling_data['soy_yield']
scaled_corn_preds, scaled_soy_preds = output_rescaling_data['corn_preds'], output_rescaling_data['soy_preds']

plt.hist(orig_soy_yield, bins=30, alpha=0.6, color='green', label='Soy Yields')
plt.hist(scaled_corn_preds, bins=30, alpha=0.6, color='orange', label='Corn -> Soy Preds')

plt.hist(orig_corn_yield, bins=30, alpha=0.6, color='red', label='Corn Yields')
plt.hist(scaled_soy_preds, bins=30, alpha=0.6, color='cyan', label='Soy -> Corn Preds')

plt.legend()
plt.xlabel('Yield Value', fontsize=24)
plt.ylabel('Frequency of Yield Value', fontsize=24)
plt.title('Histograms of Original Yields vs. Rescaled Predicted Yields', fontsize=26)
plt.show()