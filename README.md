# about

prism is a command-line tool converting between audio and image representations of frequency intensity over time.

a spectrogram is a visual representation of the frequency "spectra" of a signal over time. they are useful for a wide range of audio analysis.

this frequency-domain decomposition is somewhat natural given humans' sense of hearing is based on 

## usage

to convert a `.WAV` file into a frequency-domain, image representation, run `encode.py`:

$ python3 encode.py example.wav spectrogram.bmp

to convert a `.BMP` image into a reconstructed `.WAV` file, run `decode.py`:

$ python3 encode.py spectrogram.bmp reconstructed.wav

further command line parameters can be found by passing the `--help` flag to either python program.

# citations

[1] https://www.music.mcgill.ca/~gary/307/week6/node4.html accessed 1 nov 2024
[2] https://www.music.mcgill.ca/~gary/307/week6/node5.html accessed 1 nov 2024