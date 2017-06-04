import numpy as np

def l2_dist(m1, m2, axis=None):
    return np.sqrt(np.mean((m1 - m2)**2, axis=axis))

def l1_dist(m1, m2, axis=None):
    return np.mean(np.absolute(m1 - m2), axis=axis)

def perc_dist(m1, m2, axis=None):
    return np.mean(np.absolute(m1 - m2)/np.absolute(m1), axis=axis)