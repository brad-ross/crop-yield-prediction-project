import os
import numpy as np
import sys

MAP_PATH = os.path.expanduser('~/cs231n-satellite-images-models/saliency_maps/original_model_comparison.npz')

maps = np.load(MAP_PATH)

def visualize_maps(maps):
    corn_maps = maps['corn_maps']
    soy_maps = maps['soy_maps']
    # WIP

visualize_maps(maps)
