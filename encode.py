
# encode.py
# encode .wav file into frequency spectrogram

import click
import numpy as np
from PIL import Image
from tqdm import tqdm

from scipy.io import wavfile

RATE = 44_100
FREQ_STEP = 10

freqs = list(range(1, RATE // 2, FREQ_STEP))

WINDOW_SIZE = 8192
WINDOW_STEP = 512

@click.command()
@click.option("-i", "--in-file", required=True, help="Input mono WAV file (44,100Hz).")
@click.option("-o", "--out-file", required=True, help="Output BMP image file.")
def encode(in_file, out_file):

    rate, data = wavfile.read(in_file)

    # generate windows
    # with centers spaced WINDOW_STEP apart
    # each extending out WINDOW_SIZE / 2 in both directions
    # and tapered with a hamming window
    windows = []
    taper = np.hamming(WINDOW_SIZE)
    for w_start in range(0, len(data) - WINDOW_SIZE, WINDOW_STEP):
        w_end = w_start + WINDOW_SIZE
        window = data[w_start: w_end]
        windows.append(window)

    # test windows, 1 per frequency (with 0 and 90deg shifted options)
    test_sin = np.zeros((len(freqs), WINDOW_SIZE))
    test_cos = np.zeros((len(freqs), WINDOW_SIZE))
    t = np.linspace(0, WINDOW_SIZE / RATE, WINDOW_SIZE)

    for freq_idx, freq in enumerate(freqs):
        test_sin[freq_idx] = np.sin(2.0 * np.pi * freq * t)
        test_cos[freq_idx] = np.cos(2.0 * np.pi * freq * t)
        
        period = (1.0 / freq) * RATE
        tail = int(WINDOW_SIZE % period)

        for j in range(0, tail):
            test_sin[freq_idx][-j] = 0
            test_cos[freq_idx][-j] = 0

    spectra = []
    for window in tqdm(windows):
        spectrum = []

        magnitude_product = np.linalg.norm(test_sin) * np.linalg.norm(test_cos)
        products_sin = np.sum(np.multiply(test_sin, window) * taper, axis=1) / magnitude_product
        products_cos = np.sum(np.multiply(test_cos, window) * taper, axis=1) / magnitude_product

        spectrum = np.sqrt(products_sin ** 2 + products_cos ** 2)
        spectra.append(spectrum)

    spectra = np.array(spectra).T

    # normalize spectra to export as 0-255
    spectra = spectra * 255 / max(spectra.flatten())
    spectra = np.rint(spectra).astype(np.uint8)

    im = Image.fromarray(spectra, mode="L")
    im.save(out_file)

if __name__ == "__main__":
    encode()
