import numpy as np
import matplotlib.pyplot as plt

f = np.load('image_stats_300_ic.npz')
sizes = f['prop_nonzero_pixels']

plt.hist(sizes, 50, facecolor='#27ae60', normed=True)

plt.xlabel('Proportion')
plt.ylabel('Relative Frequency')
plt.title('Proportion of Farmland in County (n = 300)')
plt.show()

np.savetxt('image_stats_300_ic.csv', sizes, delimiter=',')
