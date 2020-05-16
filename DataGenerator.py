import numpy as np
import py_midicsv as pm

def generate_data(filename):
    csv_string = pm.midi_to_csv(filename)
    notes = np.zeros((3000, 48))
    last = 0

    for x in csv_string:
        note = x.split(',')
        # print(note)
        if note[2] == ' Note_on_c':
            time = int(note[1]) // 120
            pitch = int(note[4]) - 32
            notes[time, pitch] = 1
            if pitch < 0 or pitch > 47:
                print("ERROR")
            last = time
    notes = np.resize(notes, (last, 48))

    data = np.zeros((0, 48, 48))

    for i in range(0, notes.shape[0]-47, 4):
        data = np.append(data, np.expand_dims(notes[i:i+48, :], axis=0), axis=0)

    return data

def import_data():
    data = generate_data("midi/BienNho.mid")
    data = np.append(data, generate_data("midi/BenDoiHiuQuanh.mid"), axis=0)
    data = np.append(data, generate_data("midi/TuoiDaBuon.mid"), axis=0)
    data = np.append(data, generate_data("midi/MuaHong.mid"), axis=0)
    data = np.append(data, generate_data("midi/NhuCanhVacBay.mid"), axis=0)
    return data

