# encode.py
# encode .wav file into frequency spectrogram

# cli
import click

# analysis
import numpy as np
import scipy
from scipy import signal
from numba import jit, prange

# i/o
from scipy.io import wavfile
from PIL import Image

FREQ_MAX = 15_000


@click.command()
@click.argument("in_file", required=True)
@click.argument("out_file", required=True)
@click.option("-f", "--resample-factor", default=1, help="Resample input data before analysis.")
@click.option("-b", "--freq-bins", default=2048, help="Number of frequency bins.")
@click.option("-w", "--window-size", default=1024, help="Size of analysis windows.")
@click.option("-s", "--window-step", default=128, help="Space between analysis windows.")
def encode_wrapper(in_file, out_file, resample_factor, freq_bins, window_size, window_step):
    file_rate, data = wavfile.read(in_file)
    assert file_rate == 44100
    data = signal.resample(data, len(data) * resample_factor)
    spectra = encode(file_rate * resample_factor, data, freq_bins, window_size, window_step)
    im = Image.fromarray(spectra, mode="L")
    im.save(out_file)

@jit(nopython=True, parallel=True, nogil=True)
def encode(rate, data, freq_bins, window_size, window_step): # -> array(float64)
    freqs = (FREQ_MAX * np.linspace(0, 1, freq_bins)).astype(np.uint)
    
    # generate windows
    # with centers spaced WINDOW_STEP apart
    # each extending out WINDOW_SIZE / 2 in both directions
    # and tapered with a hamming window
    window_starts = np.arange(0, len(data) - window_size, window_step)

    windows = np.zeros((len(window_starts), window_size))
    taper = np.hamming(window_size)
    for w_idx in prange(len(windows)):
        w_start = window_starts[w_idx]
        w_end = w_start + window_size
        windows[w_idx] = data[w_start: w_end]

    # test windows, 1 per frequency (with 0 and 90deg shifted options)
    test_sin = np.zeros((len(freqs), window_size))
    test_cos = np.zeros((len(freqs), window_size))
    t = np.linspace(0, window_size / rate, window_size)
    #    phases = np.random.random((len(freqs), 2)) * 2 * np.pi

    for freq_idx in prange(len(freqs)):
        freq = freqs[freq_idx]
        test_sin[freq_idx] = 2 * freq * np.sin(2.0 * np.pi * freq * t) * taper
        test_cos[freq_idx] = 2 * freq * np.cos(2.0 * np.pi * freq * t) * taper

        test_sin[freq_idx] /= window_size
        test_cos[freq_idx] /= window_size
        
        if freq == 0:
            tail = 0
        else:
            period = (1.0 / freq) * rate
            tail = int(window_size % period)

        for j in prange(0, tail):
            test_sin[freq_idx][-j] = 0
            test_cos[freq_idx][-j] = 0

    print("dot")
    w_T = windows.T
    products_sin = np.dot(test_sin, w_T)
    products_cos = np.dot(test_cos, w_T)

    print("finalizing spectra...")
    spectra = np.sqrt(products_sin ** 2 + products_cos ** 2)
    
    # normalize spectra to export as 0-255
    spectra = spectra * 255 / max(spectra.flatten())
    spectra = np.rint(spectra).astype(np.uint8)

    return spectra

if __name__ == "__main__":
    encode_wrapper()
