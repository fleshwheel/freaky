# about

prism is a command-line tool converting between audio and image representations of frequency intensity over time.

a spectrogram is a visual representation of the frequency "spectra" of a signal over time. they are useful for a wide range of audio analysis. this tool was created to study the characteristics of PZT crystals as chaotic dynamical systems and their application in reservoir computing.

this frequency-domain decomposition is somewhat intuitive given that humans percieve sound in the frequency-intensity domain [1]. 

## usage

to convert a `.WAV` file into a frequency-domain, image representation, run `encode.py`:

$ python3 encode.py example.wav spectrogram.bmp

to convert a `.BMP` image into a reconstructed `.WAV` file, run `decode.py`:

$ python3 encode.py spectrogram.bmp reconstructed.wav

further command line parameters can be found by passing the `--help` flag to either python program.

# citations

[1] "The Effects of Noise on Man", Karl D. Kryter. Academic Press (1970). page 4.