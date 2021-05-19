#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from IPython.display import clear_output
import sys

# File name

# 1. Loading Data

# Load CSV 

par = sys.argv[1]
file = sys.argv[2]
path = "../data/raw/{file}.csv".format(file = file)

# Load CSV and remove timestamps

# 1.1 Visual
df_visual = pd.read_csv(path)
df_visual = df_visual.drop(["timestamps"], axis=1)


# 1.2 Imagery
df_imagery = pd.read_csv(path)
df_imagery = df_imagery.drop(["timestamps"], axis=1)
# df_visual_1.head()

# 2. Checking Markers

# Let's look at how the marker was generated.  Here is the format:
# 
# - [block, trial, index, task, type] <br>
# index = 1,2,3 RGB <br>
# task = perception(visual), imagery <br>
# type = Fixation, img_stim (last index) <br>

print(df_visual['Marker'].unique())


# Since we set our marker to have 4 info: #block, #trial, label, time.  We gonna split and get the class for the markers.   **Note that we shall reserve 0 for no event for raw mne, thus we shall represent class 0-9 using label 1-10.**

# 2.1 Visual

#use numpy as another view of the pandas columns for faster operation
marker_np_visual = df_visual['Marker'].values
marker_np_visual = marker_np_visual.astype(str)

for idx, marker in enumerate(marker_np_visual):
    if marker != '0':
        m = marker.split(",")
        if "Fixation" in marker:
            marker_np_visual[idx] = 0
        elif "imagery" in marker: # remove black --> visual
            marker_np_visual[idx] = 0
        elif "," in marker:
            marker_np_visual[idx] = m[2] # get classes
        else:
            marker_np_visual[idx] = 0

print(np.unique(marker_np_visual))
df_visual['Marker']= marker_np_visual.astype(int)

print(df_visual.groupby('Marker').nunique())


# 2.2 Imagery

#use numpy as another view of the pandas columns for faster operation
marker_np_imagery = df_imagery['Marker'].values
marker_np_imagery = marker_np_imagery.astype(str)

for idx, marker in enumerate(marker_np_imagery):
    if marker != '0':
        m = marker.split(",")
        if "Fixation" in marker:
            marker_np_imagery[idx] = 0
        elif "perception" in marker: # remove black --> visual
            marker_np_imagery[idx] = 0
        elif "," in marker:
            marker_np_imagery[idx] = m[2] # get classes
        else:
            marker_np_imagery[idx] = 0

print(np.unique(marker_np_imagery))
df_imagery['Marker']= marker_np_imagery.astype(int)

print(df_imagery.groupby('Marker').nunique())


# 3. Save Data

df_visual.to_pickle("../data/participants/{par}/01_Data_excel-pd/{file}_perception.pkl".format(par=par, file=file))
df_imagery.to_pickle("../data/participants/{par}/01_Data_excel-pd/{file}_imagery.pkl".format(par=par, file=file))

