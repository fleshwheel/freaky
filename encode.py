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
@click.option("-r", "--resample-factor", default=1, help="Resample input data before analysis.")
@click.option("-b", "--freq-bins", default=512, help="Number of frequency bins.")
@click.option("-w", "--window-size", default=2048, help="Size of analysis windows.")
@click.option("-s", "--window-step", default=64, help="Space between centers of consecutive analysis windows.")
@click.option("-2", "--stereo", is_flag=True)
def encode_wrapper(in_file, out_file, resample_factor, freq_bins, window_size, window_step, stereo):
    file_rate, audio = wavfile.read(in_file)
    audio = audio.T

    print(audio.dtype)
    if audio.dtype == np.int32:
        audio = audio.astype(np.float64) / (2 ** 31)
    elif audio.dtype == np.int16:
        audio = audio.astype(np.float64) / (2 ** 15)

    if len(audio.shape) > 2:
        raise Exception("unexpected WAV file shape")
    elif len(audio.shape) == 2:
        if audio.shape[0] > 2:
            print("warning: discarding all wav channels except first 2...")
            audio = audio[:2]
        if audio.shape[0] == 1: # dont know how this could happen in practice
            audio = np.concat((audio, audio))
        if not stereo:
            audio = (audio[0] + audio[1]) / 2.0
    elif len(audio.shape) == 1:
        if stereo:
            audio = np.array([audio, audio])

    if stereo:
        audio_l = signal.resample(audio[0], len(audio[0]) * resample_factor)
        audio_r = signal.resample(audio[1], len(audio[0]) * resample_factor)

        spectra_l = encode(file_rate * resample_factor, audio_l, freq_bins, window_size, window_step)
        spectra_r = encode(file_rate * resample_factor, audio_r, freq_bins, window_size, window_step)

        im = np.zeros((freq_bins, len(spectra_l[0]), 3))

        print(spectra_l.shape)

        im_r = Image.fromarray(spectra_l, mode="L")
        im_b = Image.fromarray(spectra_r, mode="L")
        im_g = Image.fromarray(spectra_l * 0, mode="L")

        Image.merge("RGB", (im_r, im_g, im_b)).save(out_file)
        
        #im[:,:,0] = spectra_l
        #im[:,:,2] = spectra_r
        #Image.fromarray(im, mode="RGB").save(out_file)
    else:
        audio = signal.resample(audio, len(audio) * resample_factor)
        spectra = encode(file_rate * resample_factor, audio, freq_bins, window_size, window_step)

        Image.fromarray(spectra, mode="L").save(out_file)

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
