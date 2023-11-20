import PySimpleGUI as sg
from pygame import mixer, time
import pygame

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

mixer.init()
is_playing = False

def verify_sound_object(audio_file):

    if not sound_path:
        sg.Popup("No song specificed.")

pygame.init()

while True:
    event, values = audio_player_window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
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