import numpy as np
import matplotlib.pyplot as plt

PLOT_COLORS = ['#27AE60', '#D91E18', '#3498DB', '#F89406']

output_rescaling_data = np.load('./poster/output_rescaling/original_model_rescaled_outputs.npz')
locs = output_rescaling_data['locs']
orig_corn_yield, orig_soy_yield = output_rescaling_data['corn_yield'], output_rescaling_data['soy_yield']
scaled_corn_preds, scaled_soy_preds = output_rescaling_data['corn_preds'], output_rescaling_data['soy_preds']

print 'correlation of soy and corn yields:', np.corrcoef(orig_soy_yield, orig_corn_yield)[0,1]

plt.hist(orig_soy_yield, bins=30, alpha=0.75, color=PLOT_COLORS[0], label='Soybean Yields')
plt.hist(scaled_corn_preds, bins=30, alpha=0.75, color=PLOT_COLORS[1], label='Corn -> Soybean Preds')

plt.hist(orig_corn_yield, bins=30, alpha=0.75, color=PLOT_COLORS[2], label='Corn Yields')
plt.hist(scaled_soy_preds, bins=30, alpha=0.75, color=PLOT_COLORS[3], label='Soybean -> Corn Preds')

#plt.legend()
plt.xlabel('Yield Value', fontsize=24)
plt.ylabel('Frequency of Yield Value', fontsize=24)
plt.show()