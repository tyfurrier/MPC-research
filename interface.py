import tkinter as tk
from playsound import playsound
import numpy as np
from scipy.io.wavfile import write
from scipy import signal
import matplotlib.pyplot as plt


def violin_sound(window: int,
                 formant_width: int = 100,
                 fundamental: int = 440):
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



if __name__ == "__main__":
    violin_sound(window=3000)