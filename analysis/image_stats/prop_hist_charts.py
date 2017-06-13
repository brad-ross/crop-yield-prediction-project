import numpy as np
import matplotlib.pyplot as plt

PLOT_COLOR = '#27ae60'

prop_300 = np.load('./poster/prop_hist/prop_300_ic.npy')

plt.xlabel('Percent of County Pixels Unmasked', fontsize=24)
plt.ylabel('Percent of Counties', fontsize=24)
plt.title('Histogram of Percent of County Pixels Unmasked', fontsize=26)
plt.hist(prop_300, bins=30, color=PLOT_COLOR, alpha=0.75, normed=True)
plt.show()