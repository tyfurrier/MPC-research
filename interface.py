import math
import tkinter as tk

import matplotlib.pyplot
import scipy.fft
import librosa
import sklearn.preprocessing
import numpy as np
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import os, shutil
from typing import Tuple
from enum import Enum
class Decomposition(Enum):
    FULL = ""
    OCTAVE = "octave_"
    HOLLOW = "missing_octave_"
    RECONSTRUCTED = "reconstructed_"

    def file_naming(self):
        if self.value == "":
            return "full"
        else:
            return self.value[:-1]

def get_cached_wave(f: int, part: Decomposition,
                    cents: int = 0) -> Tuple[np.ndarray, float]:
    """ Gets the cached wave from the given frequency and decomposition.
    Args:
        f (int): The frequency.
        part (Decomposition): Whether the get the full, octave or hollowed.
    Returns:
        Tuple[np.ndarray, int]: The wave and the sample rate.
    """
    midi_number = librosa.hz_to_midi(f) + cents / 100
    fname = os.path.join("created_samples", "cents", f"{str(midi_number).replace('.', 'p')}"
                                            f"{part.file_naming()}.wav")
    if not os.path.exists(fname):
        wave, sr = trumpet_sound(frequency=librosa.midi_to_hz(midi_number),
                                 bitrate=44100,
                                    plot=False,
                                    normalize=False,
                                 part=part
                                 )
        write(fname, sr, wave)
    return librosa.load(fname)  # todo: rename the files to be midi whole number plus cents and throw
    #  in an if cents is negative to make it the prior midi number plus complement


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
    # # Plot the STFT
    # D = np.abs(librosa.stft(audio))
    # # fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
    # fig, ax = plt.subplots()
    # img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=bitrate, y_axis='log', x_axis='time',
    #                          ax=ax)
    # fig.colorbar(img, ax=ax,
    #              # format="%+2.f dB"
    #              )


def autocorrelation(data, sr):
    ac = librosa.autocorrelate(data)[1:]
    x_ticks = [sr // (i + 1) for i in range(len(ac))]
    # for x, y in zip(x_ticks, ac):
    #     plt.plot([x], [y])
    # plt.xscale('log')
    x_ticks = [(512**i) for i in range(5)]
    plt.xticks(x_ticks, labels=x_ticks)
    print(x_ticks)
    y_ticks = [ac[sr // x] for x in x_ticks]
    data = {'x': x_ticks, 'y': ac}
    # plt.plot(x_ticks, y_ticks)
    example_data = [4096, 512, 1024, 2048]
    plt.plot(example_data, [200, -700, 700, -700])
    plt.show()


def plot_fft(data, sr):
    data = np.array(data).astype(float)
    D = np.abs(librosa.stft(data))
    plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    return D


def pure_tone_synthesizer(fundamental: int,
                          harmonic_decibels: list = None,
                          plot: bool = False,
                          bitrate: int = 44100,
                          normalize: bool = True,
                          custom_minmax: tuple = None) -> np.ndarray:
    """ Returns one second of the fundamental and its harmonics at the given decibel levels.
    The amplitudes list should include the fundamental and None for -inf decibels."""
    if harmonic_decibels is None:
        harmonic_decibels = [0]
    amplitudes = [librosa.db_to_amplitude(d) if d is not None else None for d in harmonic_decibels]
    length = 1  # seconds of pure tone to generate
    t = np.linspace(0, length, length * bitrate)
    canvas = np.zeros(bitrate)  # one second since we are using integer hz values
    pure_tone = np.sin(2 * np.pi * fundamental * t)
    for i in range(len(canvas)):
        for f, amp in enumerate(amplitudes):
            if amp is None:
                continue
            harmonic = f + 1
            # amplitude = amp * pure_tone[(((i + 1) * harmonic) - 1) % len(pure_tone)]
            amplitude = amp * np.sin(2 * np.pi * fundamental * t[i] * harmonic)
            canvas[i] += amplitude
    full_second = np.array(canvas).astype(np.float32)
    if normalize:
        return scale_numpy_wave(wave=full_second, plot=plot, minmax=custom_minmax), bitrate
    else:
        return full_second, bitrate


def trumpet_harmonic_decibels(part: Decomposition = Decomposition.FULL) -> list:
    decibels = [-22, -19, -21.5, -26,
                -25, -28, -27, -27,
                -28, -32.5, -38] \
               + [(-65 - -38) / 10 * (i + 1) - 38 for i in range(10)] \
               + [(-92 - -65) / 6 * (i + 1) - 65 for i in range(6)]  # -65 by 10k | -92 by 13k
    if part == Decomposition.FULL:
        return decibels
    elif part == Decomposition.HOLLOW:
        for i in range(1, len(decibels), 2):
            decibels[i] = None
        return decibels
    elif part == Decomposition.OCTAVE:
        return [decibels[i] if i % 2 == 1 else None for i in range(len(decibels))]
    else:
        raise ValueError(f"Unknown decomposition: {part}")


def trumpet_sound(frequency: int,
                  bitrate: int = 44100,
                  plot: bool = False,
                  normalize: bool = True,
                  part: Decomposition = Decomposition.FULL) -> np.ndarray:
    """ Returns a list of amplitudes for one second of a trumpet playing the given frequency"""
    amplitudes = [1.0,
                  0.8, 0.22, 0.4,
                  0.55
                  ]
    amplitudes = [0.36, 0.6, 0.23, 1, 0.7, 0.55, 0.18, 0.3, 0.05, 0.06]
    decibels = trumpet_harmonic_decibels(part=part)
    return pure_tone_synthesizer(fundamental=frequency,
                                 harmonic_decibels=decibels,
                                 plot=plot,
                                 bitrate=bitrate,
                                 normalize=normalize,
                                 )


def scale_numpy_wave(wave: np.ndarray, plot: bool = False,
                     minmax: tuple = None) -> np.ndarray:
    """ scales a numpy wave and returns it in int32 -2147483648 to 2147483647"""
    if minmax is None:
        min_a = np.min(wave)
        max_a = np.max(wave)
    else:
        min_a, max_a = minmax
    scaled_wave = []
    for bit in wave:
        scaled_wave.append((((bit - min_a) / (max_a - min_a)) * 2 - 1))
    if plot:
        plt.plot(scaled_wave[:1000])
        plt.show()
        plt.plot(scaled_wave[:1000])
        plt.show()
    scaled_wave = np.array(scaled_wave).astype(np.float32)
    return scaled_wave

def trumpet_missing_octave(frequency: int = 440, bitrate: int = 44100,
                           file_path: str = None,
                           overwrite_folder: bool = False):
    import simpleaudio as sa
    to_write = []

    d = trumpet_harmonic_decibels()
    octave_decibels = [d[i] if i % 2 == 1 else None for i in range(len(d))]
    missing_octave_decibels = d.copy()
    for i in range(1, len(missing_octave_decibels), 2):
        missing_octave_decibels[i] = None
    # print("octave", octave_decibels)
    # print("missing octave", missing_octave_decibels)
    # print("base decibels", d)
    check_min_max = lambda x, m: print(f"{m} Min: {np.min(x)}, {m} Max: {np.max(x)}")
    full_minmax = (-0.25772929191589355, 0.27012863755226135)
    # full_minmax = (-0.275, 0.275)  # todo I think it's precautionary to use

    missing_octave, bitrate = pure_tone_synthesizer(fundamental=frequency,
                                                    harmonic_decibels=missing_octave_decibels,
                                                    bitrate=bitrate,
                                                    plot=False,
                                                    normalize=False)
    to_write.append((missing_octave, Decomposition.HOLLOW))


    trumpet_octave, bitrate = pure_tone_synthesizer(fundamental=frequency,
                                                    harmonic_decibels=octave_decibels,
                                                    normalize=False)
    to_write.append((trumpet_octave, Decomposition.OCTAVE))

    trumpet, bitrate = trumpet_sound(frequency=frequency,
                                     normalize=False)
    to_write.append((trumpet, Decomposition.FULL))

    reconstructed = missing_octave + trumpet_octave
    to_write.append((reconstructed, Decomposition.RECONSTRUCTED))
    # check_min_max(missing_octave, "minus octave")
    # check_min_max(trumpet_octave, "octave")
    # check_min_max(reconstructed, "reconstructed")
    # check_min_max(trumpet, "original")
    plot = False
    if plot:
        plot_fft(trumpet_octave, bitrate)
        plot_fft(missing_octave, bitrate)
        plot_fft(trumpet, bitrate)
    if not np.allclose(reconstructed, trumpet):
        print(f"reconstructed and original are not close f: {frequency}")
    if file_path is None:
        file_path = "created_samples"
    if os.path.exists(file_path):
        if overwrite_folder:
            shutil.rmtree(file_path)
            os.mkdir(file_path)
    else:
        os.mkdir(file_path)
    for wave, d_ in to_write:
        fname = f"{str(round(librosa.hz_to_midi(frequency), 2)).replace('.', 'p')}" \
                f"{d_.file_naming()}.wav"
        write(os.path.join(file_path, fname), bitrate, wave)

def create_radii(radius: int = 99):
    count = 1
    p = Decomposition.FULL
    for p in [Decomposition.FULL, Decomposition.OCTAVE]:
        pure_synth, sr = get_cached_wave(f=440, cents=0, part=p)
        sum = pure_synth / count
        folder_path = os.path.join("created_samples", f"440_spread_{p.file_naming()}")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
        for i in range(1, radius):
            new_count = count + 2
            sum = sum * count/new_count
            count = new_count
            left, _ = get_cached_wave(f=440, cents=i, part=p)
            right, _ = get_cached_wave(f=440, cents=-i, part=p)
            sum += left / count
            sum += right / count
            write(os.path.join(folder_path, f"{i}_ct.wav"), sr, sum)

def tone_pair_osc_trumpet(
    cents: int,
    f: int = 440,
    folder_name: str = "pairs",
    clear_dir: bool = True
):
    folder_path = os.path.join("created_samples",
                               folder_name)
    if os.path.exists(folder_path):
        if clear_dir:
            shutil.rmtree(folder_path)
            os.mkdir(folder_path)
    else:
        os.mkdir(folder_path)
    for p in [
        Decomposition.FULL,
        # Decomposition.OCTAVE,
        # Decomposition.HOLLOW
              ]:
        bitrate = 44100
        fname = f"{cents}_{p.file_naming()}.wav"
        sum = None
        for c in [cents, -cents]:
            freq = librosa.midi_to_hz(librosa.hz_to_midi(f) + c / 100)
            wave, sr = trumpet_sound(frequency=freq,
                                     bitrate=bitrate,
                                     plot=False,
                                     normalize=False
                                     )
            sum = wave if sum is None else sum + wave
            write(os.path.join(folder_path, fname), int(sr), sum)


if __name__ == "__main__":
    # fft_of_file(os.path.join(
    #     # os.path.pardir,
    #     "external_samples",
    #     "violin_solo.wav"))
    # create_radii(radius=50)
    test, sr = trumpet_sound(frequency=440.1, bitrate=44100, plot=False, normalize=False)
    write(os.path.join("created_samples", "trumpet_pure_440+1.wav"), sr, test)
    if False:
        tone_pair_osc_trumpet(cents=0, clear_dir=True)
        for i in range(1, 20):
            tone_pair_osc_trumpet(cents=i/10.0, clear_dir=False)
        for i in range(1, 100, 4):
            tone_pair_osc_trumpet(cents=i, clear_dir=False)
    if False:
        for i in range(-100, 100):
            midi_number = librosa.hz_to_midi(440) + (i/100)
            print(f"midi number: {midi_number}")
            trumpet_missing_octave(frequency=librosa.midi_to_hz(midi_number),
                                   file_path=os.path.join("created_samples", "cents"))
    # old_generation()
    # D = np.abs(librosa.stft(audio, n_fft))
    # plt.figure()
    # librosa.display.specshow(D, sr=sr, y_axis='log', x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    # plt.show()
    # playsound(os.path.join(
    #     # os.path.pardir,
    #     "external_samples",
    #     "violin_solo.wav"))
