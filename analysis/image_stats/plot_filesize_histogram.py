import numpy as np
import matplotlib.pyplot as plt

f = np.load('image_stats_1000_filesize.npz')
sizes = f['sizes']

plt.hist(sizes, 50, facecolor='green')

plt.xlabel('Sizes')
plt.ylabel('Frequency')
plt.title('Distribution of Image Sizes (n = 1000)')
plt.show()
