import numpy as np
import matplotlib.pyplot as plt

file = np.load('image_stats.npz')
#bins, edges = np.histogram(np.maximum(file['widths'], file['heights']), 50)
bins, edges = np.histogram(file['aspect_ratios'], 50)
left,right = edges[:-1],edges[1:]
X = np.array([left,right]).T.flatten()
Y = np.array([bins,bins]).T.flatten()

plt.plot(X,Y)
plt.xlabel('Aspect ratio')
plt.ylabel('Frequency')
plt.title('Histogram of aspect ratio')
plt.savefig('aspect_ratio_histogram.png')
