#!/usr/bin/env python3

import numpy as np
from PIL import Image
from numba import jit, prange

def average_entropy(filename, stereo=False):
    im = Image.open(filename)

    if stereo:
        im = im.convert("RGB")
        r, g, b = im.split()

        return (_average_entropy(np.asarray(g, dtype=np.float64)),
                _average_entropy(np.asarray(r, dtype=np.float64)))

    else:
        im = im.convert("L")
        return _average_entropy(np.asarray(im, dtype=np.float64))

def average_cross_entropy(filename, flip=True):
    im = Image.open(filename).convert("RGB")
    r, g, b = im.split()
    if flip:
        r, g = g, r
    return _average_cross_entropy(np.asarray(g, dtype=np.float64),
                                  np.asarray(r, dtype=np.float64))

@jit(nopython=True, parallel=True, nogil=True)
def _average_entropy(im):
    num_freqs = im.shape[1]
    num_spectra = im.shape[0]
    im = im ** 2 # to account for sqrt scaling

    entropies = np.zeros(num_spectra)
    for row_idx in prange(num_spectra):
        psd = (im[row_idx] ** 2) / num_freqs

        pdf = psd
        if np.sum(psd) != 0:
            pdf /= np.sum(psd)

        entropy = 0
        for p in pdf:
            if p != 0:
                entropy -= p * np.log(p)

        entropies[row_idx] = entropy
        
    return sum(entropies) / len(entropies)

@jit(nopython=True, parallel=True, nogil=True)
def _average_cross_entropy(im1, im2):
    num_freqs = im1.shape[1]
    num_spectra = im2.shape[0]
    im1 = im1 ** 2 # to account for sqrt scaling
    im2 = im2 ** 2

    entropies = np.zeros(num_spectra)
    for row_idx in prange(num_spectra):
        psd1 = (im1[row_idx] ** 2) / num_freqs
        psd2 = (im2[row_idx] ** 2) / num_freqs

        pdf1, pdf2 = psd1, psd2
        if np.sum(psd1) != 0:
            pdf1 /= np.sum(psd1)
        if np.sum(psd2) != 0:
            pdf2 /= np.sum(psd2)

        entropy = 0
        for p1, p2 in zip(pdf1, pdf2):
            if p2 != 0:
                entropy -= p1 * np.log(p2)

        entropies[row_idx] = entropy
        
    return sum(entropies) / len(entropies)
