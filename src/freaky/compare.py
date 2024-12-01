#!/usr/bin/env python3

import numpy as np
from PIL import Image

def imgdiff(file1, file2):
    im1 = np.asarray(Image.open(file1).convert("RGB"), dtype=np.float64) / 255
    im2 = np.asarray(Image.open(file2).convert("RGB"), dtype=np.float64) / 255

    if im1.shape != im2.shape:
        print("warning! cannot compare images of differing shapes.")
        return -1
    
    return np.mean(np.abs(im1 - im2).flatten())

    
    
