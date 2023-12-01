from interface import spread_file_naming, OUTPUT_DIR, Decomposition, get_spread_sound
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_stimuli(cents: float,
                      decomposition: Decomposition):
    """
    Visualize the stimuli for a given spread and decomposition.
    :param cents: The radius in each formant in cents DO NOT DO MORE THAN TWO DECIMALS.
    :param decomposition: The decomposition.
    """
    sound, sr = get_spread_sound(folder_name="radius_50",
                                 cents=cents,
                                 part=decomposition)
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(0)

    dt = 1 / sr  # sampling interval (used to be 0.1)
    Fs = sr  # sampling frequency
    length_in_seconds = len(sound) / sr
    t = np.arange(0, length_in_seconds, dt)

    s = sound  # the signal

    title_suffix = f"for {cents} cents spread"

    fig = plt.figure(figsize=(7, 7), layout='constrained')
    axs = fig.subplot_mosaic([["signal", "signal"],
                              ["magnitude", "log_magnitude"],
                              ["phase", "angle"]])

    # plot time signal:
    axs["signal"].set_title(f"Signal {title_suffix}")
    axs["signal"].plot(t, s, color='C0')
    axs["signal"].set_xlabel("Time (s)")
    axs["signal"].set_ylabel("Amplitude")

    # plot different spectrum types:
    axs["magnitude"].set_title(f"Magnitude Spectrum {title_suffix}")
    axs["magnitude"].magnitude_spectrum(s, Fs=Fs, color='C1')

    axs["log_magnitude"].set_title(f"Log. Magnitude Spectrum {title_suffix}")
    axs["log_magnitude"].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

    axs["phase"].set_title(f"Phase Spectrum {title_suffix}")
    axs["phase"].phase_spectrum(s, Fs=Fs, color='C2')

    axs["angle"].set_title("Angle Spectrum")
    axs["angle"].angle_spectrum(s, Fs=Fs, color='C2')

    plt.show()

def results_df():
    df = pd.read_csv("spreadsheet.csv")
    df.reset_index(inplace=True)
    df["ratio"] = np.maximum(df["sound1"] / df["sound2"], df["sound2"] / df["sound1"])
    df["different"] = df["decision"]
    df["different"].replace(["Same", "Different"], [0, 1], inplace=True)
    df["size"] = 1
    df.head()
    print(df)
    fig = px.scatter(df, x="ratio", y="different", color="subject",
                     hover_data=["sound1", "sound2", "decision"])
    fig.show()
    return df

if __name__ == "__main__":
    df = results_df()
    print(df.where(df["ratio"] < 1.5).count())
    if False:
        for cents in [0.1, 0.5, 1, 2, 5, 10, 25, 40, 49.9]:
            visualize_stimuli(cents=cents, decomposition=Decomposition.FULL)