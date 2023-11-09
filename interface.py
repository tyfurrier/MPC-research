import tkinter as tk

import scipy.fft
import librosa
from playsound import playsound
import numpy as np
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import os


def fft_of_file(fname: str):
    """ Plots the fft of the given file.
    Args:
        fname (str): The file name.
    """
    # spectrogram = scipy.signal.spectrogram(data, bitrate)
    import librosa

    # Load the audio file
    audio, sr = librosa.load(fname)

    # Create the spectrogram
    spectrogram = librosa.stft(audio)
    plot_fft(data=audio, sr=sr)

    plot_autocorrelation(data=audio, sr=sr)
    # # Plot the STFT
    # D = np.abs(librosa.stft(audio))
    # # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=bitrate, y_axis='log', x_axis='time',
    #                          ax=ax)
    # fig.colorbar(img, ax=ax,
    #              # format="%+2.f dB"
    #              )

def plot_autocorrelation(data, sr):
    ac = librosa.autocorrelate(data)
    plt.figure()
    librosa.display.specshow(ac, sr=sr, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()


def plot_fft(data, sr):
    D = np.abs(librosa.stft(data))
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    return D



def violin_sound(window: int,
                 formant_width: int = 100,
                 fundamental: int = 256):
    """ Plays a violin sound with the duration of the given window size in milliseconds.
    Args:
        window (int): The duration in milliseconds.
        formant_width (int): The width of the formant in cents. The percieved magnitude at the formant will be the same
        but the actual magnitude at fine points in the formant will be different. A formant width of 0 will result in a
        pure tone at the fundamental. A formant width of 100 should result in frequencies at the f plus and minus 1 and
        in between.
        fundamental (int): The fundamental frequency in Hz. Defaults to 440.
        """
    db_levels = [60, 50, 45, 50, 43, 48, 37, 48, 30, 20]
    equation = """
    db = 20log(y)
    db = log(y^20)
    10^db = y^20
    10^(db/20) = y"""
    to_magnitude = lambda db: 10 ** (db / 20)
    partials_magnitude = [to_magnitude(db) for db in db_levels]
    partials_magnitude = db_levels
    bitrate = 44100
    master_vol = 0.0000001
    freq = 440
    length = window // 1000

    # create time values
    t = np.linspace(0, length, length * bitrate, dtype=np.float32)
    # generate y values for signal
    y = partials_magnitude[0] * master_vol * np.sin(2 * np.pi * fundamental * t)
    for harmonic, level in enumerate(partials_magnitude[1:]):
        harmonic += 1
        frequency = fundamental * harmonic + fundamental
        y += level * master_vol * np.sin(2 * np.pi * frequency * t)
        print(f"Harmonic {harmonic} at {frequency} Hz with magnitude {level}")

    # save to wave file
    write("violin.wav", bitrate, y)
    return y, bitrate

def violin_sound_v2(window: int,
                 formant_width: int = 100,
                 fundamental: int = 256):
    """ Plays a violin sound with the duration of the given window size in milliseconds.
    Args:
        window (int): The duration in milliseconds.
        formant_width (int): The width of the formant in cents. The percieved magnitude at the formant will be the same
        but the actual magnitude at fine points in the formant will be different. A formant width of 0 will result in a
        pure tone at the fundamental. A formant width of 100 should result in frequencies at the f plus and minus 1 and
        in between.
        fundamental (int): The fundamental frequency in Hz. Defaults to 440.
        """

    bitrate = 44100
    master_vol = 0.0000001
    freq = 440
    length = window // 1000

    # create time values
    t = np.linspace(0, length, length * bitrate, dtype=np.float32)
    magnitudes = []
    sample_data, sample_sr = librosa.load(os.path.join('external_samples', 'violin_solo.wav'))
    for f in range(1, 31):
        magnitudes.append(librosa.autocorrelate(sample_data, f=(f*fundamental)))
    components = []
    for f, level in enumerate(magnitudes):
        frequency = fundamental * (f + 1)
        components.append(level * np.sin(2 * np.pi * frequency * t))
        print(f"Harmonic {f + 1} at {frequency} Hz with magnitude {level}")
    final_wave = np.sum(components, axis=0)

    # save to wave file
    write("violin.wav", bitrate, final_wave)
    return final_wave, bitrate



if __name__ == "__main__":
    fft_of_file(os.path.join(
        # os.path.pardir,
        "external_samples",
        "violin_solo.wav"))
    # our_version, our_bitrate = violin_sound_v2(window=3000)
    # plot_fft(data=our_version, sr=our_bitrate)
    # audio, sr = librosa.load(os.path.join("external_samples", "violin_solo.wav"))
    # D = np.abs(librosa.stft(audio, n_fft))
    # plt.figure()
    # librosa.display.specshow(D, sr=sr, y_axis='log', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    # playsound("violin.wav")