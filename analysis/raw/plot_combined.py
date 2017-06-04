import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

raw = np.load('raw_2013_1_1.npz')
raw_image_slice = raw['raw_2013_t31_b6']

vis = np.load('../vis/smap_vis_2013_1_1.npz')
vis_image_slice = vis['vis_soy_t31_b6']

raw_img = Image.fromarray(raw_image_slice).convert("RGBA")
vis_img = Image.fromarray(vis_image_slice).convert("RGBA")

raw_img.paste(vis_img, (0, 0), vis_img)

raw_img.show()

