import sys
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


BLOCK_SIZE = 1024
MIN_PERIOD = 1


window = np.hamming(BLOCK_SIZE * 3)

rate, data = wavfile.read("440.wav")

chunks = []

for i in range(0, len(data), BLOCK_SIZE):
    chunks.append(data[i : i  + BLOCK_SIZE])

chunks = chunks[: -1]

periods = np.arange(MIN_PERIOD, BLOCK_SIZE // 2) / rate

t = np.linspace(0, 1, BLOCK_SIZE)


triple_t = np.linspace(0, 3, BLOCK_SIZE * 3)
components = [np.sin(2 * np.pi * triple_t / p) for p in periods]

magnitudes = np.zeros((len(chunks), len(periods)))

for chunk_idx, chunk in tqdm(list(enumerate(chunks[1:-1]))):
    for component_idx, component in enumerate(components):
        full_chunk = np.append(np.append(chunks[chunk_idx - 1], chunk), chunks[chunk_idx + 1])
        total = 0

        
        for i in range(1, 256, 31): # must be >0 bc of list slicing lol
            total += abs(np.dot(component[i:], full_chunk[:-i]))
        magnitudes[chunk_idx][component_idx] = abs(total / 32)

magnitudes /= max(magnitudes.flatten())
        
print(magnitudes)

print(components[0])

#plt.plot(components[0])
#plt.plot(components[1])
#plt.plot(components[2])

#plt.plot(magnitudes[:,
#plt.show()
plt.imshow(magnitudes, cmap='gray')
plt.show()

print("periods range from")
print(rate * min(periods))
print(rate * max(periods))
#p sample / (rate samples / second)) 
