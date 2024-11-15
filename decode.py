import click
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from numba import jit, prange

from PIL import Image
from scipy.signal import resample
from scipy.io import wavfile

#FREQ_MAX = SAMPLE_RATE / 2
FREQ_MAX = 20_000


@click.command()
@click.argument("in_file", required=True)
@click.argument("out_file", required=True)
@click.option("-r", "--sample-rate", default=44100, help="Sample rate of output audio WAV file.")
@click.option("-f", "--resample-factor", default=1, help="Resample input data before analysis.")
@click.option("-w", "--window-length", default=64, help="Space between centers of consecutive analysis windows.")
def decode_wrapper(in_file, out_file, sample_rate, resample_factor, window_length):
    image_data = np.asarray(Image.open(in_file)).T.astype(np.float64)

    if len(image_data.shape) == 3:
        # raise error if its something other than rgb
        assert image_data.shape[0] == 3
        print("notice: summing rgb channels to achieve greyscale...")
        image_data = image_data.sum(axis=0) / 3
    
    spectra = image_data / 255
    result = decode(spectra, sample_rate * resample_factor, window_length)
    result = resample(result, len(result) // resample_factor)
    wavfile.write(out_file, sample_rate, result)

@jit(nopython=True, parallel=True, nogil=True)
def decode(spectra, sample_rate, window_length):

    num_windows = spectra.shape[0]
    num_freqs = spectra.shape[1]
    num_samples = num_windows * window_length
    
    freqs = np.linspace(5, FREQ_MAX, num_freqs)
    
    spectra = spectra * spectra
#    spectra = np.exp(spectra)

    length = num_windows * window_length
    T = np.linspace(0, length / sample_rate, num_windows * window_length)

    result = np.zeros(num_samples)

    #components = np.zeros((len(freqs), length))
    phases = np.random.random(len(freqs)) * sample_rate * 4

    print(spectra.shape)
    
    for i_f in range(len(freqs)):
        term = np.zeros(num_samples)
        
        component = freqs[i_f] * T
        if freqs[i_f] != 0:
            component += phases[i_f] / freqs[i_f]
        component = np.sin(2 * np.pi * component)

        last_coeff = spectra[0][i_f]
        for i_w in prange(num_windows):
            coeff = spectra[i_w][i_f]
            window = np.linspace(last_coeff, coeff, window_length)
            wstart = i_w * window_length
            wend = wstart + window_length
            term[wstart: wend] = np.multiply(window, component[wstart: wend])
            last_coeff = coeff
        result += term
  
    result /= max(np.abs(result.flatten()))
    
    return result
    

if __name__ == "__main__":
    decode_wrapper()
