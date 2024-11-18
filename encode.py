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

FREQ_MAX = 20_000

@click.command()
@click.argument("in_file", required=True)
@click.argument("out_file", required=True)
@click.option("-x", "--resample-factor", default=1, help="Resample input data before analysis.")
@click.option("-b", "--freq-bins", default=512, help="Number of frequency bins.")
@click.option("-w", "--window-size", default=2048, help="Size of analysis windows.")
@click.option("-s", "--window-step", default=64, help="Space between centers of consecutive analysis windows.")
def encode_wrapper(in_file, out_file, resample_factor, freq_bins, window_size, window_step):
    file_rate, data = wavfile.read(in_file)
    if data.dtype == np.int32:
        data = data.astype(np.float64) / (2 ** 31)
    #assert file_rate == 44100
    data = signal.resample(data, len(data) * resample_factor)
    spectra = encode(file_rate * resample_factor, data, freq_bins, window_size, window_step)
    im = Image.fromarray(spectra, mode="L")
    im.save(out_file)

@jit(nopython=True, parallel=True, nogil=True)
def encode(rate, data, freq_bins, window_size, window_step): # -> array(float64)
    freqs = np.linspace(0, FREQ_MAX, freq_bins)

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
    test = np.zeros((len(freqs), window_size)).astype(np.complex128)
#    test_cos = np.zeros((len(freqs), window_size))
    t = np.linspace(0, window_size / rate, window_size)
    
    for freq_idx in range(len(freqs)):
        freq = freqs[freq_idx]
        tp1 = t #+ phases[freq_idx][0]
        tp2 = t #+ phases[freq_idx][1]
        test[freq_idx] = np.cos(2 * np.pi * freq * tp1) * taper + np.sin(2 * np.pi * freq * tp2) * taper * 1j

    w_T = windows.T.astype(np.complex128)
    products = np.dot(test, w_T)
    
    print("finalizing spectra...")

    spectra = np.abs(products) / len(windows)

    spectra = spectra #/ max(spectra.flatten())

    spectra = np.sqrt(spectra)
    
    spectra = spectra * 255 * 4
    
    spectra = spectra.astype(np.uint8)


    return spectra

if __name__ == "__main__":
    encode_wrapper()
