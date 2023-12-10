#!/usr/bin/python3

import sys
import numpy as np
from PIL import Image
from scipy.io import wavfile

BLOCK_SIZE = 256
assert BLOCK_SIZE % 2 == 0

def make_blocks(data):
    overlap = int(BLOCK_SIZE / 2)
    return np.array([data[i : i + BLOCK_SIZE] for i in range(0, len(data) - BLOCK_SIZE, overlap)])

with open("440.wav", "rb") as f:
    rate, data = wavfile.read(f)
    
blocks = make_blocks(data)

window = np.hamming(BLOCK_SIZE)
freqs = np.fft.fftfreq(BLOCK_SIZE)
result = np.zeros((len(blocks), len(freqs)))
for idx, block in enumerate(blocks):
    windowed_block = np.multiply(block, window)
    w = np.fft.fft(windowed_block)
    result[idx] = abs(w)

print("first results like")
print("result[0] = ", result[0])
    
result = result / max(result.flatten())
result = (result * (2 ** 8)).astype(np.int8)

result = np.rot90(result)

print(result.shape)

im = Image.fromarray(result, mode = "L")

im.save("out.bmp")

im = np.asarray(Image.open("out.bmp"))
im = np.rot90(im, 3).astype(np.float32) / (2 ** 8)

print("shape of u")
print(im.shape)

import matplotlib.pyplot as plt

magnitudes = np.zeros((im.shape[0] * int(BLOCK_SIZE / 2), BLOCK_SIZE))

for (line_idx, line) in enumerate(im):
    for idx in range(int(BLOCK_SIZE / 2)):
        magnitudes[line_idx * int(BLOCK_SIZE / 2) + idx] = line

magnitudes = magnitudes * 1_552_737.0

print("new results like")
print("magnitudes[0] = ", magnitudes[0])

freqs = np.fft.fftfreq(BLOCK_SIZE, 1 / rate)


num_samples = len(magnitudes)
final_length = num_samples / rate # final length in seconds

# i add the "f +" term to spac eit out phase wise
full_spectrum = np.array([[np.sin(f + (f * t * 2 * np.pi / rate)) for f in freqs] for t in np.linspace(0, final_length, num_samples)])
result = np.zeros(im.shape[0] * int(BLOCK_SIZE / 2)).astype(np.float32)

print("setting this many")

for i in range(len(result)):
    result[i] = np.dot(full_spectrum[i], magnitudes[i])

result /= np.max(result, axis=0)
    
with open("translated.wav", "wb") as f:
    wavfile.write(f, rate, result)

print(result.shape)
