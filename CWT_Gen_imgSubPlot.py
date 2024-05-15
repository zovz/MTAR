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

# Create Labels, rampup is pre-labled
normal_data['label'] = "normal"
vibration_data['label'] = "vibration"
friction_data['label'] = "friction"

# Create a CWT of the first sequence
dt = 0.001  # 1000 Hz sampling
fs = 1/dt
scale = np.arange(1,1001) / fs # normalize
scale = pywt.frequency2scale('cmor1.5-1.0', scale)

def save_mul_image(iter, seq2, scale, baseString, grid = [2,3]):
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

    saveString = "CWTImages\\" + baseString + str(iter) + ".png"

    # save the figure as a png file
    plt.subplots_adjust(wspace=0, hspace=0)
    try:
        plt.savefig(saveString, bbox_inches='tight', pad_inches=0)
    except:
        # Create the folder
        os.makedirs(os.path.dirname(saveString))
        plt.savefig(saveString, bbox_inches='tight', pad_inches=0)
    plt.close()

# Create images for Friction
seq_ax_m_friction = []
seq_ay_m_friction = []
seq_az_m_friction = []
seq_ax_b_friction = []
seq_ay_b_friction = []
seq_az_b_friction = []

for i in range(0,int(len(friction_data)/SEQUENCE_LENGTH)):
    seq = []
    for values in friction_data['ax_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_m_friction.append(seq)
    seq = []

    for values in friction_data['ay_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_m_friction.append(seq)
    seq = []

    for values in friction_data['az_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_m_friction.append(seq)
    seq = []

    for values in friction_data['ax_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_b_friction.append(seq)
    seq = []

    for values in friction_data['ay_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_b_friction.append(seq)
    seq = []

    for values in friction_data['az_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_b_friction.append(seq)

    

# Create images for Normal
seq_ax_m_normal = []
seq_ay_m_normal = []
seq_az_m_normal = []
seq_ax_b_normal = []
seq_ay_b_normal = []
seq_az_b_normal = []

for i in range(0,int(len(normal_data)/SEQUENCE_LENGTH)):
    seq = []
    for values in normal_data['ax_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_m_normal.append(seq)
    seq = []

    for values in normal_data['ay_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_m_normal.append(seq)
    seq = []

    for values in normal_data['az_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_m_normal.append(seq)
    seq = []

    for values in normal_data['ax_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_b_normal.append(seq)
    seq = []

    for values in normal_data['ay_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_b_normal.append(seq)
    seq = []

    for values in normal_data['az_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_b_normal.append(seq)




# Create images for Vibration
seq_ax_m_vibration = []
seq_ay_m_vibration = []
seq_az_m_vibration = []
seq_ax_b_vibration = []
seq_ay_b_vibration = []
seq_az_b_vibration = []


for i in range(0,int(len(vibration_data)/SEQUENCE_LENGTH)):
    seq = []
    for values in vibration_data['ax_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_m_vibration.append(seq)
    seq = []

    for values in vibration_data['ay_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_m_vibration.append(seq)
    seq = []

    for values in vibration_data['az_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_m_vibration.append(seq)
    seq = []

    for values in vibration_data['ax_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_b_vibration.append(seq)
    seq = []

    for values in vibration_data['ay_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_b_vibration.append(seq)
    seq = []

    for values in vibration_data['az_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_b_vibration.append(seq)


    

#all_rampup_data = rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'v']
all_rampup_data = rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'ax_motor']
#all_rampdown_data = rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'v']
all_rampdown_data = rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'ax_motor']

# Create images for RampUp
seq_ax_m_rampup = []
seq_ay_m_rampup = []
seq_az_m_rampup = []
seq_ax_b_rampup = []
seq_ay_b_rampup = []
seq_az_b_rampup = []

for i in range(0,int(len(all_rampup_data)/SEQUENCE_LENGTH)):
    seq = []
    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'ax_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_m_rampup.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'ay_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_m_rampup.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'az_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_m_rampup.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'ax_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_b_rampup.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'ay_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_b_rampup.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampup", 'az_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_b_rampup.append(seq)

# Create images for RampDown
seq_ax_m_rampdown = []
seq_ay_m_rampdown = []
seq_az_m_rampdown = []
seq_ax_b_rampdown = []
seq_ay_b_rampdown = []
seq_az_b_rampdown = []

for i in range(0,int(len(rampupdown_data)/SEQUENCE_LENGTH)):
    seq = []
    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'ax_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_m_rampdown.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'ay_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_m_rampdown.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'az_motor'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_m_rampdown.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'ax_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ax_b_rampdown.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'ay_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_ay_b_rampdown.append(seq)
    seq = []

    for values in rampupdown_data.loc[rampupdown_data['label'] == "rampdown", 'az_bearing'][i*SEQUENCE_LENGTH:(i+1)*SEQUENCE_LENGTH]:
        seq.append(values)
    seq_az_b_rampdown.append(seq)


# rampup_data_seq = []
# rampdown_data_seq = []
# sequence_rampup = []
# sequence_rampdown = []
# for values in all_rampup_data:
#     sequence_rampup.append(values)
#     if len(sequence_rampup) == SEQUENCE_LENGTH:
#         rampup_data_seq.append(sequence_rampup)
#         sequence_rampup = []

# for values in all_rampdown_data:
#     sequence_rampdown.append(values)
#     if len(sequence_rampdown) == SEQUENCE_LENGTH:
#         rampdown_data_seq.append(sequence_rampdown)
#         sequence_rampdown = []
    
friction_data_seq = lambda i: [seq_ax_m_friction[i], seq_ay_m_friction[i], seq_az_m_friction[i], seq_ax_b_friction[i], seq_ay_b_friction[i], seq_az_b_friction[i]]
normal_data_seq = lambda i: [seq_ax_m_normal[i], seq_ay_m_normal[i], seq_az_m_normal[i], seq_ax_b_normal[i], seq_ay_b_normal[i], seq_az_b_normal[i]]
vibration_data_seq = lambda i: [seq_ax_m_vibration[i], seq_ay_m_vibration[i], seq_az_m_vibration[i], seq_ax_b_vibration[i], seq_ay_b_vibration[i], seq_az_b_vibration[i]]
rampup_data_seq = lambda i: [seq_ax_m_rampup[i], seq_ay_m_rampup[i], seq_az_m_rampup[i], seq_ax_b_rampup[i], seq_ay_b_rampup[i], seq_az_b_rampup[i]]
rampdown_data_seq = lambda i: [seq_ax_m_rampdown[i], seq_ay_m_rampdown[i], seq_az_m_rampdown[i], seq_ax_b_rampdown[i], seq_ay_b_rampdown[i], seq_az_b_rampdown[i]]

# friction_data_seq = [seq_ax_m_friction, seq_ay_m_friction, seq_az_m_friction, seq_ax_b_friction, seq_ay_b_friction, seq_az_b_friction]
# normal_data_seq = [seq_ax_m_normal, seq_ay_m_normal, seq_az_m_normal, seq_ax_b_normal, seq_ay_b_normal, seq_az_b_normal]
# vibration_data_seq = [seq_ax_m_vibration, seq_ay_m_vibration, seq_az_m_vibration, seq_ax_b_vibration, seq_ay_b_vibration, seq_az_b_vibration]
# rampup_data_seq = [seq_ax_m_rampup, seq_ay_m_rampup, seq_az_m_rampup, seq_ax_b_rampup, seq_ay_b_rampup, seq_az_b_rampup]
# rampdown_data_seq = [seq_ax_m_rampdown, seq_ay_m_rampdown, seq_az_m_rampdown, seq_ax_b_rampdown, seq_ay_b_rampdown, seq_az_b_rampdown]


# Create a pool of processes
paths = ["Friction\\Friction_", "Normal\\Normal_", "Vibration\\Vibration_", "RampUp\\RampUp_", "RampDown\\RampDown_"]
data = [friction_data_seq, normal_data_seq, vibration_data_seq, rampup_data_seq, rampdown_data_seq] 
data_len = [len(seq_ax_m_friction), len(seq_ax_m_normal), len(seq_ax_m_vibration), len(seq_ax_m_rampup), len(seq_ax_m_rampdown)]
#data = lambda i: [friction_data_seq[i], normal_data_seq[i], vibration_data_seq[i], rampup_data_seq[i], rampdown_data_seq[i]]


if __name__ == '__main__':
    # for nr in range(len(data)):
    #     print("Creating images for " + paths[nr].split("\\")[0] + "...")
    #     #[(i, data[nr](i), scale, paths[nr], [2,3]) for i in range(data_len[nr])]
    #     with multiprocessing.Pool() as pool:
    #         # Use starmap to apply save_image to each sequence in friction_data_seq
    #         pool.starmap(save_mul_image, [(i, data[nr](i), scale, paths[nr], [2,3]) for i in range(data_len[nr])])
    #     # Wait for process to finish
    #     pool.close()

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