# analyze.py
# convert .wav file into spectrogram

import sys
import numpy as np
from PIL import Image

from scipy.io import wavfile
import matplotlib.pyplot as plt

#from PIL import Image

RATE = 44_100
FREQ_STEP = 100

freqs = list(range(1, RATE // 2, FREQ_STEP))

WINDOW_SIZE = 2048
WINDOW_STEP = 1024

rate, data = wavfile.read(sys.argv[1])
print(rate)
print(len(data))

# generate windows
windows = []
for w_start in range(0, len(data), WINDOW_SIZE):
    w_end = w_start + WINDOW_SIZE
    window = data[w_start: w_end]
    if len(window) < WINDOW_SIZE:
        print("padding!")
        window = np.append(window, np.repeat(0, WINDOW_SIZE - len(window)))
    windows.append(window)

# test windows, 1 per frequency (with 0 and 90deg shifted options)
test_windows = []
t = np.linspace(0, WINDOW_SIZE / RATE, WINDOW_SIZE)
for freq in freqs:
    test_windows.append((np.sin(2 * np.pi * freq * t),
                         np.cos(2 * np.pi * freq * t)))    

spectra = []
for window in windows:
    spectrum = []
    for (tw_sin, tw_cos) in test_windows:
        cdot, sdot = np.dot(tw_cos, window), np.dot(tw_sin, window)
        spectrum.append(max((abs(cdot), abs(sdot))))
    spectra.append(spectrum)


spectra = np.array(spectra).T

# normalize spectra to export as 0-255
spectra = spectra * 255 / max(spectra.flatten())
spectra = np.rint(spectra).astype(np.uint8)
print(spectra)

plt.imshow(spectra)
plt.show()
im = Image.fromarray(spectra, mode="L")
im.show()
im.save("out.bmp")
