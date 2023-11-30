import PySimpleGUI as sg
from pygame import mixer, time
import pygame
import pyaudio
import wave
import sys

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



mixer.init()
is_playing = False

layout= [
    [sg.Text("Choose a file: "), sg.Input(), sg.FileBrowse(key="-SOUND_PATH-")],
    [sg.Text(size=(12,1), key='-STATUS-')],
    [
        sg.Button('Play', pad=(10, 0), key='-PLAY-'),
        sg.Button('Pause', pad=(10, 0), key='-PAUSE-'),
        sg.Button('Stop', pad=(10, 0), key='-STOP-'),
        sg.Slider(range=(0, 100),
                orientation='h', size=(50, 20), enable_events=True, key = '-VOLUME-', default_value= 100)
    ]
]

audio_player_window = sg.Window('Audio Player', layout, finalize=True)
def verify_sound_object(audio_file):

    if not sound_path:
        sg.Popup("No song specificed.")

pygame.init()

while True:
    event, values = audio_player_window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break

    sound_path = values["-SOUND_PATH-"]
    if not sound_path:
        sg.Popup("No song specificed.")
        continue
    song = mixer.Sound(sound_path)

    song_length = song.get_length()
    song_channel = mixer.Channel(2)

    if event == '-PLAY-':
        audio_player_window['-STATUS-'].update('Playing')
        is_playing = True
        print(song_channel.play(song))
    elif event == '-PAUSE-':
        if not is_playing:
            audio_player_window['-STATUS-'].update('Playing')
            is_playing = True
            song_channel.unpause()
        else:
            audio_player_window['-STATUS-'].update('Paused')
            is_playing = False
            song_channel.pause()
    elif event == '-STOP-':
        audio_player_window['-STATUS-'].update('Stopped')
        song_channel.stop()
        is_playing = False
    elif event == '-VOLUME-':
        volume = values['-VOLUME-']
        song_channel.set_volume(volume/100)

audio_player_window.close()