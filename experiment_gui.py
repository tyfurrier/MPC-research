import PySimpleGUI as sg
from pygame import mixer, time
import pygame
import pyaudio
import wave
import sys
import random
import os
from pandas import DataFrame
import datetime
from interface import OUTPUT_DIR, spread_file_naming, Decomposition




def play(file_path: str):
    # length of data to read.
    chunk = 1024

    # validation. If a wave file hasn't been specified, exit.

    '''
    ************************************************************************
          This is the start of the "minimum needed to read a wave"
    ************************************************************************
    '''
    # open the file for reading.
    wf = wave.open(sys.argv[1], 'rb')

    # create an audio object
    p = pyaudio.PyAudio()

    # open stream based on the wave object which has been input.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while data:
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)


    # cleanup stuff.
    wf.close()
    stream.close()
    p.terminate()

def add_to_csv(row: dict):
    df = DataFrame(row)
    df.to_csv("spreadsheet.csv", mode='a', header=False)

mixer.init()
is_playing = False
PARTICIPANT_KEY = "-PARTICIPANT-"
layout= [
    [sg.Text("Your name"), sg.Input(key=PARTICIPANT_KEY)],
    [sg.Text("Sound 1"), sg.Button('Play', pad=(10, 0), key='-PLAY1-'),
     sg.Text("Sound 2"), sg.Button('Play', pad=(10, 0), key='-PLAY2-'),],
    [sg.Text(size=(12,1), key='-STATUS-')],
    [
        sg.Button('Same', pad=(10, 0), key='-SAME-'),
        sg.Button('Different', pad=(10, 0), key='-DIFFERENT-'),
        sg.Button('Next', pad=(10, 0), key='-STOP-'),
        sg.Slider(range=(0, 100),
                  orientation='h', size=(50, 20), enable_events=True, key='-VOLUME-',
                  default_value=100)
    ]
]

audio_player_window = sg.Window('Audio Player', layout, finalize=True,
                                return_keyboard_events=True)

pygame.init()


def get_random_sound(force_spread: int = None):

    spread = random.randint(0, 499) if force_spread is None else force_spread
    return os.path.join(OUTPUT_DIR, "radius_50", spread_file_naming(radius=spread/10,
                                                                    part=Decomposition.FULL,
                                                                    ))

def get_comparison() -> tuple:
    return get_random_sound(), get_random_sound()


COMPARISON_ROW = None
tracking = {}
actions_string = ""
while True:
    event, values = audio_player_window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    if COMPARISON_ROW is None:
        sound_one, sound_two = get_comparison()
        COMPARISON_ROW = {"time_started": [pygame.time.get_ticks()],
                          "est_start": [datetime.datetime.now()],
                          "time_ended": [],
                          "sound1": [sound_one.split("full_")[1].split("_c")[0]],
                          "sound2": [sound_two.split("full_")[1].split("_c")[0]],
                          "participant": [values[PARTICIPANT_KEY]],
                          "behavior": [],
                          "status": [],
                          }
    COMPARISON_ROW['participant'] = [values[PARTICIPANT_KEY]]
    COMPARISON_ROW['status'] = [audio_player_window['-STATUS-'].get()]
    first = mixer.Sound(sound_one)
    second = mixer.Sound(sound_two)

    song_channel = mixer.Channel(2)

    if event in ['-PLAY1-', '1']:
        actions_string += "-p1-"
        if tracking.get('-STATUS2-') == 'Playing':
            second.stop()
            tracking['-STATUS2-'].update('Stopped')
        print('playing 1')
        first.play()
        time.wait(1000)
        print("2 done")
    elif event in ['-PLAY2-', '2']:
        print("playing 2")
        actions_string += "-p2-"
        if tracking.get('-STATUS1-') == 'Playing':
            first.stop()
            tracking['-STATUS1-'].update('Stopped')
        second.play()
        time.wait(1000)
        print("2 done")
    elif event == '-SAME-':
        actions_string += "-s-"
        audio_player_window['-STATUS-'].update('Same')
    elif event == '-DIFFERENT-':
        actions_string += "-d-"
        audio_player_window['-STATUS-'].update('Different')
    elif event == '-STOP-':
        if COMPARISON_ROW["participant"][0] in [None, ""]:
            sg.popup_error("Please write your name first")
            continue
        if audio_player_window['-STATUS-'].get() == "":
            sg.popup_error("Please select a response")
            continue
        COMPARISON_ROW['time_ended'].append(pygame.time.get_ticks())
        COMPARISON_ROW['behavior'].append(actions_string)
        COMPARISON_ROW['status'] = [audio_player_window['-STATUS-'].get()]
        print(COMPARISON_ROW)
        add_to_csv(COMPARISON_ROW)
        COMPARISON_ROW = None
        actions_string = ""
        audio_player_window['-STATUS-'].update('')
        print("ON TO NEW COMPARISON")
    elif event == '-VOLUME-':
        volume = values['-VOLUME-']
        song_channel.set_volume(volume/100)

audio_player_window.close()