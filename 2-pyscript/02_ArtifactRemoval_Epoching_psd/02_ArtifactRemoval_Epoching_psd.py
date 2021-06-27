#!/usr/bin/env python
# coding: utf-8

# ## 02 Artifact_Removal
# NOTE: this notebook includes psd method. The outputs are "logged" and the removal of irrelevant frequencies and psd were done before epoching.
# 
# drift: 0.01-1 Hz <br>
# no_drift: 1-40 Hz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle
from mne import Epochs, find_events
from IPython.display import clear_output
import sys
from utils import *

# 1. Loading Data

# 1.1 Load pd

par = sys.argv[1]
file = sys.argv[2]
fmin = float(sys.argv[3])
fmax = float(sys.argv[4])
tmin = float(sys.argv[5])
tmax = float(sys.argv[6])
electrode_zone = sys.argv[7]
electrodes = [i for i in sys.argv[8].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()]
psd_enable = sys.argv[9]

# Here I decided to keep stuff in a form of dictionary where **keys** indicate the **task and time**

path_task_types= ["_perception", "_imagery"]
# Just wanted to make the names look nice

task_types = ["perception", "imagery"]

dfs_task_types = {}

for i,path_task_type in enumerate(path_task_types):
    path = "../data/participants/{par}/01_Data_excel-pd/{file}{path_task_type}.pkl".format(par = par, file = file, path_task_type = path_task_type)
    f = open(path,'rb')
    df = pickle.load(f)
    print(df['Marker'].unique())
    dfs_task_types["{task_types}".format(task_types = task_types[i])] = df


# 2. Artifact Removal & Power Spectrum Density

# To be installed!!!!!!!!! <br>
# **pip3 install matplotlib>=3.0.3**

# Artifacts that are restricted to a narrow frequency range can sometimes be repaired by filtering the data. Two examples of frequency-restricted artifacts are slow drifts and power line noise. Here we illustrate how each of these can be repaired by filtering.
# 
# But first we gonna use Python MNE as it provides many useful methods for achieving these tasks.  So first, we gonna transform our pandas to mne type.  Here is the function transforming df to raw mne.

import mne
from mne import create_info
from mne.io import RawArray


# Transform df to raw mne.
raws = {}
for task_type in task_types:
    print(f"========================= {task_type} ==========================")
    raws[task_type] = df_to_raw(dfs_task_types[task_type], electrodes)


# Independent component analysis according to the selected electrode configuration
from mne.preprocessing import ICA
for task_type in task_types:
    ica = ICA(n_components=len(electrodes), random_state=32)
    # print(raws[task_type].shape)
    ica.fit(raws[task_type])
    #ica.apply(raws[task_type])


event_id = {'0': 1, '1' : 2, '2': 3}

import time
for task_type in task_types:
    X=[]
    picks= mne.pick_types(raws[task_type].info, eeg=True)
    epochs = getEpochs(raws[task_type], event_id, tmin, tmax, picks)
    y = epochs.events[:, -1]
    y = y - 1
    for epoch in epochs.iter_evoked():
        # clear_output(wait=True)
#         epoch.plot()
#         time.sleep(2)
        epoch_copy = epoch.copy()
        # Filter Frequency
        epoch_copy.filter(fmin, fmax, method='iir')
        # Min-Max Normalization
        epoch_copy = epoch_copy.data
        min = epoch_copy.min()
        max = epoch_copy.max()
        epoch_copy = (epoch_copy - min) / (max-min)
        print(epoch_copy.shape)
        if psd_enable is True:
            psd,_,_,_ = get_psd(epoch_copy,electrodes)
#           psd = psd.mean(axis=1)
            X.append(psd)
        else:
            X.append(epoch_copy)
    X = np.array(X)

    if psd_enable is True:
        np.save("../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task_type}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_psd_X".format(par=par,file=file, fmin = fmin, fmax=fmax,task_type = task_type, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), X)
        np.save("../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task_type}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_psd_y".format(par=par,file=file, fmin = fmin, fmax=fmax,task_type = task_type, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), y)
    else:
        np.save("../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task_type}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_X".format(par=par,file=file, fmin = fmin, fmax=fmax,task_type = task_type, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), X)
        np.save("../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task_type}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_y".format(par=par,file=file, fmin = fmin, fmax=fmax,task_type = task_type, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), y)

print(X.shape)
print(y.shape)