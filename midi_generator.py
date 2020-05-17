from midiutil import MIDIFile
import pygame

def play_img(inp):

    MyMIDI = MIDIFile()  # One track, defaults to format 1 (tempo track is created
    # automatically)
    MyMIDI.addTempo(0, 0, 100)
    for time, x in enumerate(inp):
        for pitch, y in enumerate(x):
            if y == 1:
                MyMIDI.addNote(0, 0, pitch+16, time*0.125, 1, 100)

    with open("output.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

    print("Start")

    pygame.init()
    pygame.mixer.music.load("output.mid")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(1000)
