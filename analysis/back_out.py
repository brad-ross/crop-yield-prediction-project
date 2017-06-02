import numpy as np

def freq_to_intens(dhist, val, range_min=0, range_max=255):
    """
    dhist: a histogram, used for getting the range of frequencies
    val: the gradient value to map to an intensity

    Returns the mapped value.
    """
    max_val = np.max(dhist)
    min_val = np.min(dhist)
    val = np.fmax(np.fmin(val, max_val), min_val) # Clamp value to be in range
    return np.interp(val, [min_val, max_val], [range_min, range_max])

def back_out_single(img, dhist, bin_seq=None):
    """
    img: a numpy array of shape (H, W) representing a single band
    dhist: an array of shape (32,) containing gradients
    bin_seq: the boundaries of the histogram buckets

    Returns the backed-out image in the form of a numpy array
    """
    if bin_seq is None:
        bin_seq = np.linspace(1, 4999, 33)
    out = np.zeros_like(img)
    for i in range(len(bin_seq) - 1):
        start, end = bin_seq[i], bin_seq[i + 1]
        if i != len(bin_seq) - 2: # not the last bucket
            out[(start <= img) * (img < end)] = dhist[i]
        else: # last bucket, <= vs <
            out[(start <= img) * (img <= end)] = dhist[i]
    return out

def back_out_multiple(imgs, dhists, bin_seq=None):
    """
    imgs: a numpy array of shape (N, H, W) representing a collection of N images
    dhist: an array of shape (N, 32) containing gradients
    bin_seq: the boundaries of the histogram buckets

    Returns the backed-out images in the form of a numpy array
    """
    if bin_seq is None:
        bin_seq = np.linspace(1, 4999, 33)
    outs = np.zeros_like(imgs)
    for n in range(imgs.shape[0]):
        outs[n] = back_out_single(imgs[n], dhists[n])
    return outs
