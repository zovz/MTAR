import pandas as pd
import pywt
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize

import multiprocessing
import matplotlib.pyplot as plt
import pywt

import os
import shutil
import random



# Load the datasets
with open("Normal_120s.pkl", "rb") as f:
    normal_data = pd.read_pickle(f)

with open("Vibration_120s.pkl", "rb") as f:
    vibration_data = pd.read_pickle(f)

# with open("RampUpDown_120s.pkl", "rb") as f:
#     rampupdown_data = pd.read_pickle(f)
    
with open("RampUpDown_300s_Labelled.pkl", "rb") as f:
    rampupdown_data = pd.read_pickle(f)

with open("Friction_120s.pkl", "rb") as f:
    friction_data = pd.read_pickle(f)

def calculate_distance(df):
    df['v_motor'] = (df['ax_motor']**2 + df['ay_motor']**2 + df['az_motor']**2)**0.5
    df['v_bearing'] = (df['ax_bearing']**2 + df['ay_bearing']**2 + df['az_bearing']**2)**0.5
    df['v'] = (df['v_motor'] + df['v_bearing'])
    return df

normal_data = calculate_distance(normal_data)
vibration_data = calculate_distance(vibration_data)
friction_data = calculate_distance(friction_data)
rampupdown_data = calculate_distance(rampupdown_data)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


SEQUENCE_LENGTH = 120

# Create Labels
normal_data['label'] = "normal"
vibration_data['label'] = "vibration"
# for i, label in enumerate(rampupdown_data['label']):
#     if label == 0:
#         rampupdown_data['label'].iloc[i] = "rampdown"
#     else:
#         rampupdown_data['label'].iloc[i] = "rampup"
# Create seperate dataset for rampup and rampdown
# rampup_data = rampupdown_data[rampupdown_data['label'] == "rampup"]
# rampdown_data = rampupdown_data[rampupdown_data['label'] == "rampdown"]
friction_data['label'] = "friction"

# Create a CWT of the first sequence
dt = 0.001  # 1000 Hz sampling
fs = 1/dt
scale = np.arange(1,1001) / fs # normalize
scale = pywt.frequency2scale('cmor1.5-1.0', scale)


def save_image(i, seq, scale, baseString):
    # Create a CWT of the sequence
    cwt, freqs = pywt.cwt(seq, scale, 'mexh')
    plt.imshow(cwt, aspect='auto', extent=[0, 1, 1, 1000], cmap='gist_ncar')
    plt.axis('off')

    saveString = "CWTImages\\" + baseString + str(i) + ".png"
    #"CWTImages\\Friction\\Friction_"
    # save the figure as a png file
    try:
        plt.savefig(saveString, bbox_inches='tight', pad_inches=0)
    except:
        # Create the folder
        os.makedirs(os.path.dirname(saveString))
    plt.close()

def save_mul_image(iter, seq2, scale, grid = [2,3]):
    # Create a CWT of the sequence
    cwts = []
    for seq in seq2:
        cwt, freqs = pywt.cwt(seq, scale, 'mexh')
        cwts.append(cwt)

    fig, axs = plt.subplots(grid[0], grid[1], figsize=(20, 20))
    for i in range(grid[0]*grid[1]):
        #create a grid of images
        axs[i//3, i%3].imshow(cwts[i], aspect='auto', extent=[0, 1, 1, 1000], cmap='gist_ncar')
        axs[i//3, i%3].axis('off')

    saveString = "CWTImages\\Friction\\Friction_" + str(iter) + ".png"

    # save the figure as a png file
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(saveString, bbox_inches='tight', pad_inches=0)
    plt.close()

# Create images for Friction
friction_data_seq = []
# Create sequence of images for normal data
# for i in range(0,int(len(friction_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in friction_data['v'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     friction_data_seq.append(sequence)

for i in range(0,int(len(friction_data)/SEQUENCE_LENGTH)):
    sequence = []
    for values in friction_data['ax_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        sequence.append(values)
    friction_data_seq.append(sequence)

# for i in range(0,int(len(friction_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in friction_data['ay_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     friction_data_seq.append(sequence)

# for i in range(0,int(len(friction_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in friction_data['az_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     friction_data_seq.append(sequence)


# Create images for Normal
normal_data_seq = []
# for i in range(0,int(len(normal_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in normal_data['v'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     normal_data_seq.append(sequence)

for i in range(0,int(len(normal_data)/SEQUENCE_LENGTH)):
    sequence = []
    for values in normal_data['ax_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        sequence.append(values)
    normal_data_seq.append(sequence)

# for i in range(0,int(len(normal_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in normal_data['ay_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     normal_data_seq.append(sequence)

# for i in range(0,int(len(normal_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in normal_data['az_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     normal_data_seq.append(sequence)


# Create images for Vibration
vibration_data_seq = []
# for i in range(0,int(len(vibration_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in vibration_data['v'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     vibration_data_seq.append(sequence)

for i in range(0,int(len(vibration_data)/SEQUENCE_LENGTH)):
    sequence = []
    for values in vibration_data['ax_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        sequence.append(values)
    vibration_data_seq.append(sequence)

# for i in range(0,int(len(vibration_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in vibration_data['ay_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     vibration_data_seq.append(sequence)

# for i in range(0,int(len(vibration_data)/SEQUENCE_LENGTH)):
#     sequence = []
#     for values in vibration_data['az_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
#         sequence.append(values)
#     vibration_data_seq.append(sequence)
    

#all_rampup_data = rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'v']
all_rampup_data = rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'ax_motor']
#all_rampdown_data = rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'v']
all_rampdown_data = rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'ax_motor']

rampup_data_seq = []
rampdown_data_seq = []
sequence_rampup = []
sequence_rampdown = []
for values in all_rampup_data:
    sequence_rampup.append(values)
    if len(sequence_rampup) == SEQUENCE_LENGTH:
        rampup_data_seq.append(sequence_rampup)
        sequence_rampup = []

for values in all_rampdown_data:
    sequence_rampdown.append(values)
    if len(sequence_rampdown) == SEQUENCE_LENGTH:
        rampdown_data_seq.append(sequence_rampdown)
        sequence_rampdown = []


# Create a pool of processes
paths = ["Friction\\Friction_", "Normal\\Normal_", "Vibration\\Vibration_", "RampUp\\RampUp_", "RampDown\\RampDown_"]
data = [friction_data_seq, normal_data_seq, vibration_data_seq, rampup_data_seq, rampdown_data_seq]

if __name__ == '__main__':
    for nr in range(0, 5):
        print("Creating images for " + paths[nr].split("\\")[0] + "...")
        print([(i, seq, scale, paths[nr]) for i, seq in enumerate(data[nr])])
        with multiprocessing.Pool() as pool:
            # Use starmap to apply save_image to each sequence in friction_data_seq
            pool.starmap(save_image, [(i, seq, scale, paths[nr]) for i, seq in enumerate(data[nr])])
        # Wait for process to finish
        pool.close()

    #Move 20% of the images to the test folder

    for path in paths:
        files = os.listdir("CWTImages\\"+path.split("\\")[0])
        random.shuffle(files)
        for i in range(0, int(len(files) * 0.2)):
            try:
                shutil.move("CWTImages\\" + path.split("\\")[0] + "\\" + files[i], "CWTImagesTest\\"+ path.split("\\")[0] + "\\" + files[i])
            except:
                os.makedirs(os.path.dirname("CWTImagesTest\\"+ path.split("\\")[0] + "\\" + files[i]))
                shutil.move("CWTImages\\" + path.split("\\")[0] + "\\" + files[i], "CWTImagesTest\\"+ path.split("\\")[0] + "\\" + files[i])