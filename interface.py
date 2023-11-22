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

def get_cached_wave(f: int, part: Decomposition) -> Tuple[np.ndarray, float]:
    """ Gets the cached wave from the given frequency and decomposition.
    Args:
        f (int): The frequency.
        part (Decomposition): Whether the get the full, octave or hollowed.
    Returns:
        Tuple[np.ndarray, int]: The wave and the sample rate.
    """
    fname = os.path.join("created_samples", f"trumpet_pure_{part.value}{f}.wav")
    return librosa.load(fname)


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
            amplitude = amp * pure_tone[(i * harmonic) % len(pure_tone)]
            canvas[i] += amplitude
    full_second = np.array(canvas).astype(np.float32)
    if normalize:
        return scale_numpy_wave(wave=full_second, plot=plot, minmax=custom_minmax), bitrate
    else:
        return full_second, bitrate


def trumpet_harmonic_decibels():
    decibels = [-22, -19, -21.5, -26,
                -25, -28, -27, -27,
                -28, -32.5, -38] \
               + [(-65 - -38) / 10 * (i + 1) - 38 for i in range(10)] \
               + [(-92 - -65) / 6 * (i + 1) - 65 for i in range(6)]  # -65 by 10k | -92 by 13k
    return decibels


def trumpet_sound(frequency: int,
                  bitrate: int = 44100,
                  plot: bool = False,
                  normalize: bool = True,) -> np.ndarray:
    """ Returns a list of amplitudes for one second of a trumpet playing the given frequency"""
    amplitudes = [1.0,
                  0.8, 0.22, 0.4,
                  0.55
                  ]
    amplitudes = [0.36, 0.6, 0.23, 1, 0.7, 0.55, 0.18, 0.3, 0.05, 0.06]
    decibels = trumpet_harmonic_decibels()
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


def violin_sound_v3(window: int,
                 formant_width: int = 100,
                 fundamental: int = 256,
                    fname="violin.wav",
                    plot=True):
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
    length = window // 1000

    # create time values
    t = np.linspace(0, length, length * bitrate, dtype=np.float32)
    magnitudes = []
    sample_data, sample_sr = librosa.load(os.path.join('external_samples', 'violin_solo.wav'))
    stft = librosa.stft(sample_data, n_fft=bitrate, win_length=bitrate)
    # stft = plot_fft(data=sample_data, sr=sample_sr)
    collection_radius = 0.5
    collection_frequency_up = librosa.midi_to_hz(librosa.hz_to_midi(fundamental) + collection_radius) - fundamental
    collection_frequency_down = fundamental - librosa.midi_to_hz(librosa.hz_to_midi(fundamental) - collection_radius)

    components = []
    for f in range(1, bitrate//2//fundamental - 1):
    # for f in range(1, 2):
        frequency = fundamental * f
        upper_bound = math.ceil(frequency + collection_frequency_up * f)
        lower_bound = math.floor(frequency - collection_frequency_down * f) # todo: multiply collection_freq_down by f
        if f > 20:
            if frequency > 8300:
                drop_off = 60
            else:
                drop_off = (frequency - 4200) / 4100 * 60 + magnitudes[19]
            base_db = magnitudes[0]
            magnitudes.append(base_db - drop_off)
            continue
        semi_down = math.floor(librosa.midi_to_hz(librosa.hz_to_midi(frequency) - 1))
        semi_up = math.ceil(librosa.midi_to_hz(librosa.hz_to_midi(frequency) + 1))
        magnitude_data = []
        for hz in range(lower_bound, upper_bound):
            if hz >= stft.shape[0]:
                break
            magnitude_data.append(np.average(stft[hz]))  # todo: this hz isn't doing anything, try dividing by 4. need to map to the right frequency
        magnitudes.append(np.average(magnitude_data))  # switch sum/average
    for f, level in enumerate(magnitudes):
        frequency = fundamental * (f + 1)
        db_level = level
        level = librosa.db_to_amplitude(level)
        #print(f"Harmonic {f + 1} at {frequency} Hz with amplitude {level} based on db of {db_level}")

        if formant_width != 0:  # this is where we widen partials
            components_of_partial =[]
            local_width = formant_width * f
            local_level = level/(local_width*2)
            for hz in range(frequency - local_width, frequency + local_width):
                components_of_partial.append(local_level * np.sin(2 * np.pi * frequency * t))
            components.append(np.sum(components_of_partial, axis=0))
        else:
            components.append(level * np.sin(2 * np.pi * frequency * t))

    final_wave = np.sum(components, axis=0).astype(np.float32)
    scaled_wave = scale_numpy_wave(final_wave, plot=plot)

    # save to wave file
    # if os.path.exists("created_samples"):
    #     shutil.rmtree("created_samples")
    # os.mkdir("created_samples")

    write(os.path.join("created_samples", fname), bitrate, scaled_wave)
    if plot:
        ret_data, ret_sr = librosa.load(fname)
        plt.plot(ret_data[:1000])
        plt.show()
    return scaled_wave, bitrate

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

def create_radii(radius: int = 30):
    count = 1
    p = Decomposition.FULL
    for p in Decomposition:
        pure_synth, sr = get_cached_wave(f=440, part=p)
        sum = pure_synth / count
        folder_path = os.path.join("created_samples", f"440_spread_{p.file_naming()}")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.mkdir(folder_path)
        for i in range(1, radius):
            left, _ = get_cached_wave(f=440 - i, part=p)
            right, _ = get_cached_wave(f=440 + i, part=p)
            sum += left / count
            sum += right / count
            write(os.path.join(folder_path, f"{i}_hz.wav"), sr, sum)



if __name__ == "__main__":
    # fft_of_file(os.path.join(
    #     # os.path.pardir,
    #     "external_samples",
    #     "violin_solo.wav"))
    # create_radii()
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
