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

# 1. Loading Data

# 1.1 Load pd

par = sys.argv[1]
file = sys.argv[2]
fmin = float(sys.argv[3])
fmax = float(sys.argv[4])


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

def df_to_raw(df):
    sfreq = 125
    ch_names = list(df.columns)
    ch_types = ['eeg'] * (len(df.columns) - 1) + ['stim']
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')

    df = df.T  #mne looks at the tranpose() format
    df[:-1] *= 1e-6  #convert from uVolts to Volts (mne assumes Volts data)

    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    raw = mne.io.RawArray(df, info)
    raw.set_montage(ten_twenty_montage)

    #try plotting the raw data of its power spectral density
    raw.plot_psd()

    return raw


# Transform df to raw mne.

raws = {}
for task_type in task_types:
    print(f"========================= {task_type} ==========================")
    raws[task_type] = df_to_raw(dfs_task_types[task_type])

def getEpochs(raw, event_id, tmin, tmax, picks):

    #epoching
    events = find_events(raw)
    
    #reject_criteria = dict(mag=4000e-15,     # 4000 fT
    #                       grad=4000e-13,    # 4000 fT/cm
    #                       eeg=100e-6,       # 150 μV
    #                       eog=250e-6)       # 250 μV

    reject_criteria = dict(eeg=100e-6)  #most voltage in this range is not brain components

    epochs = Epochs(raw, events=events, event_id=event_id, 
                    tmin=tmin, tmax=tmax, baseline=None, preload=True,verbose=False, picks=picks)  #8 channels
    print('sample drop %: ', (1 - len(epochs.events)/len(events)) * 100)

    return epochs

def get_psd(raw, filter=True):
    '''
    return log-transformed power spectra density, freq, mean and std 
    '''
    raw_copy = raw.copy()
    if(filter):
        raw_copy.filter(fmin, fmax, method='iir')
        # if drift == "drift":
        #     raw_copy.filter(fmin, fmax, method='iir')
        # else:
        #     raw_copy.filter(1, 40, method='iir')
#             raw_copy.plot_psd()     
    psd, freq = mne.time_frequency.psd_welch(raw_copy,n_fft = 96, verbose=False)
    psd =  np.log10(psd)
    mean = psd.mean(0)
    std = psd.std(0)
    return psd, freq, mean, std
#     return raw_copy

def plot_psd(raw):
    psd, freq, mean, std = get_psd(raw)
    fig, ax = plt.subplots(figsize=(10,5))
    for i in range(8):
        ax.plot(freq,psd[i] ,label=raw.info['ch_names'][i], lw=1, alpha=0.6)
    ax.fill_between(250//2, mean - std, mean + std, color='k', alpha=.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitube (dBV)')
    ax.set_title('EEG of ')
    ax.legend()
    plt.show()

event_id = {'0': 1, '1' : 2, '2': 3}
tmin = 0.115 #0
tmax = 0.875
import time
for task_type in task_types:
    X=[]
    picks= mne.pick_types(raws[task_type].info, eeg=True)
    epochs = getEpochs(raws[task_type], event_id, tmin, tmax, picks)
    y = epochs.events[:, -1]
    y = y - 1
    for epoch in epochs.iter_evoked():
        clear_output(wait=True)
#         epoch.plot()
#         time.sleep(2)
        psd,_,_,_ = get_psd(epoch, filter=True)
#         psd = psd.mean(axis=1)
        X.append(psd)
    X = np.array(X)
    np.save("../data/participants/{par}/02_ArtifactRemoval_Epoching_psd/{file}_{task_type}_{fmin}_{fmax}_X".format(par=par,file=file, fmin = fmin, fmax=fmax,task_type = task_type), X)
    np.save("../data/participants/{par}/02_ArtifactRemoval_Epoching_psd/{file}_{task_type}_{fmin}_{fmax}_y".format(par=par,file=file, fmin = fmin, fmax=fmax, task_type = task_type), y)

print(X.shape)
print(y.shape)

