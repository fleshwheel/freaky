import click
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.io import wavfile


RATE = 44_100
# aka WINDOW_STEP
WINDOW_SIZE = 4096
FREQ_STEP = 20


@click.command()
@click.option("-i", "--in-file", required=True, help="Input BMP file.")
@click.option("-o", "--out-file", required=True, help="Output mono WAV file (44,100Hz).")
def decode(in_file, out_file):
    spectra = np.asarray(Image.open(in_file)).T.astype(np.float64) / 255

    length = spectra.shape[0] * WINDOW_SIZE

    T = np.linspace(0, length / RATE, spectra.shape[0] * WINDOW_SIZE)

    freqs = list(range(20, RATE // 2, FREQ_STEP))

    components = np.zeros((len(freqs), length))

    phases = np.random.random(len(freqs)) * 2 * np.pi
    for i, (freq, phase) in enumerate(zip(freqs, phases)):
        components[i] = np.sin(2 * np.pi * freq * T + phase)

    chunks = []
    for window_idx in range(spectra.shape[0]):
        # turn spectrum sample into a column vector
        spec_col = np.array(spectra[window_idx], ndmin=2).T
        chunk = np.tile(spec_col, WINDOW_SIZE)
        chunks.append(chunk)
        
    coeffs = np.hstack(chunks)
        
    result = coeffs * components

    result = sum(components)
    result /= max(result.flatten())

    wavfile.write(out_file, RATE, result)

if __name__ == "__main__":
    decode()
