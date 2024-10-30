import click

import sys
import numpy as np
from PIL import Image
from scipy.io import wavfile

RATE = 44_100
WINDOW_SIZE = 2048
FREQ_STEP = 100

spectra = np.asarray(Image.open(sys.argv[1])).T.astype(np.float64) / 255

length = spectra.shape[0] * WINDOW_SIZE

print(spectra.shape)
print(length / RATE)

T = np.linspace(0, length / RATE, spectra.shape[0] * WINDOW_SIZE)

freqs = list(range(1, RATE // 2, FREQ_STEP))

components = []

for freq in freqs:
    components.append(np.sin(2 * np.pi * freq * T))

for spec_idx, spectrum in enumerate(spectra):
    for freq_idx, freq in enumerate(freqs):
        for i in range(WINDOW_SIZE):
            components[freq_idx][spec_idx * WINDOW_SIZE + i] *= spectrum[freq_idx]

result = sum(components)
result /= max(result.flatten())

wavfile.write("decoded.wav", RATE, result)
