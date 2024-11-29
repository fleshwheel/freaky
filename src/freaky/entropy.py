#!/usr/bin/env python3

import numpy as np
from PIL import Image
from numba import jit, prange

@jit(nopython=True, parallel=True, nogil=True)
def average_entropy(filename):
    # no support for RGB yet
    im = np.asarray(Image.open(filename).convert("L")).T

    num_freqs = im.shape[1]
    num_spectra = im.shape[0]
    im = im ** 2 # to account for sqrt scaling

    entropies = np.zeros(num_spectra)
    for row_idx in range(num_spectra):
        psd = (im[row_idx] ** 2) / num_freqs

        pdf = psd
        if np.sum(psd) != 0:
            pdf /= np.sum(psd)

        entropy = 0
        for p in pdf:
            if p != 0:
                entropy -= p * np.log(p)

        entropies[row_idx] = entropy
        
    return sum(entropies)

                    
