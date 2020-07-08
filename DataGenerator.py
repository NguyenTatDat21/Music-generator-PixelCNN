import numpy as np
import py_midicsv as pm
import os
import matplotlib.pyplot as plt


def generate_data(filename):
    csv_string = pm.midi_to_csv(filename)
    notes = np.zeros((60000, 96))
    last = 0
    start = False
    start_time = 0
    for x in csv_string:
        note = x.split(',')
        if note[2] == ' Note_on_c':
            if not start:
                start_time = int(note[1]) // 60
            start = True
            time = int(note[1]) // 60 - start_time
            pitch = int(note[4]) - 16
            if pitch <= 0:
                print("ERROR")
            notes[time, pitch] = 1
            last = time
    notes = np.resize(notes, (last, 96))

    data = np.zeros((0, 48, 96))

    for i in range(0, notes.shape[0] - 47, 40):
        if np.sum(notes[i:i + 48, :]) > 15:
            data = np.append(data, np.expand_dims(notes[i:i + 48, :], axis=0), axis=0)
    return data


def import_data(dir_name='./midi/'):
    data = np.zeros((0, 48, 96))
    for st in os.listdir(dir_name):
        data = np.append(data, generate_data(dir_name + st), axis=0)
    return data


