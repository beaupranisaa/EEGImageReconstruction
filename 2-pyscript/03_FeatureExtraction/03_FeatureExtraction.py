#!/usr/bin/env python
# coding: utf-8

# ## 03 Feature Extraction
#%%

# Imports

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import fnmatch
import yaml
import sys
from utils import *
from models import EEGEncoder
import csv
import os.path
np.set_printoptions(threshold=sys.maxsize)

#%%

# Check GPU availability
# from chosen_gpu import get_freer_gpu
device = torch.device(get_freer_gpu()) 
print("Configured device: ", device)

#%%

# 1. Loading Data
par = sys.argv[1]
file = sys.argv[2]
fmin = float(sys.argv[3])
fmax = float(sys.argv[4])
task = sys.argv[5]
electrode_zone = sys.argv[6]
electrodes = [i for i in sys.argv[7].replace('[', ' ').replace(']', ' ').replace(',', ' ').split()]
model_name = "cnn"
roundno = sys.argv[8]
tmin = float(sys.argv[9])
tmax = float(sys.argv[10])
psd_enable = sys.argv[11]


print("#############Configuration#################")
print("par:", par)
print("round:", roundno)
print("file:", file)
print("fmin", fmin)
print("fmax", fmax) 
print("tmin", tmin)
print("tmax", tmax) 
print("task", task)
print("psd", psd_enable)
print("electrode_zone:", electrode_zone)
print("electrodes", electrodes)
print("##############################")

#%%

if psd_enable is True:
    X_ = np.load('../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_psd_X.npy'.format(par=par,file=file, task=task,fmin = fmin, fmax = fmax, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), allow_pickle=True)
    y = np.load('../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_psd_y.npy'.format(par=par,file=file, task=task,fmin = fmin, fmax = fmax, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), allow_pickle=True)

else:
    X_ = np.load('../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_X.npy'.format(par=par,file=file, task=task,fmin = fmin, fmax = fmax, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), allow_pickle=True)
    y = np.load('../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_y.npy'.format(par=par,file=file, task=task,fmin = fmin, fmax = fmax, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), allow_pickle=True)

X = X_ # get_electrode(X_,electrodes)

# 1.1 Check shape
# [# stim, # electrod, # datapoint]

print(X.shape)
print(y.shape)


# 1.2 Plot
# ### Plot to see wheter eegs have drift or not
# print(X[100][0].shape)
# fig, ax = plt.subplots(16,1,figsize=(20,50),sharex=True)

# for i in range(data.shape[1]):
#     ax[i].plot(X[100][i])


# 2. Reserve data for Real TEST
# - test_size: 0.
# - 10% of data is reserved for the real test --> X_test, y_test
# - 90% will be again divided into (train,test,val) --> X_model, y_model

# <img src="img/Split.jpg" width=300 height=300 />

# 2.1 Reserve some data for TEST
from sklearn.model_selection import train_test_split

X_trainval, X_test, y_trainval, y_test = train_test_split( X, y, test_size=0.1, stratify= y)
#X_model, X_real_test, y_model, y_real_test = train_test_split( X, y, test_size=0.1, random_state=42, stratify= y)

# Check if number of classes is equal
check_split(X_trainval, X_test, y_trainval, y_test, 'trainval', 'test')

# 2.2 Train Val Split
X_train, X_val, y_train, y_val = train_test_split( X_trainval, y_trainval, test_size=0.33, stratify= y_trainval)
check_split(X_train, X_val, y_train, y_val,'train','val')

# 3. Prepare Train Val Test Data 

# - 10 can be thought of as totally new eeg records and will be used as the real evaluation of our model.
# - For X : Chunking eeg to lengh of 10 data point in each stimuli's eeg
# - For y(lebels) : Filled the lebels in y because we chunk X ( 1 stimuli into 6 chunk). We have 500 labels before but now we need 500 x 6 = 3000 labels

# 3.1 Chunking Train Set

chunk_size = 10

print('=================== X train ==================')
print(f'Oringinal X shape {X_trainval.shape}')
X_train = chunk_data(X_train, chunk_size)
print(f'Chunked X : {X_train.shape}') # (#stim, #chunks, #electrodes, #datapoint per chunk)
chunk_per_stim = X_train.shape[1]
X_train = X_train.reshape(-1,len(electrodes),chunk_size)
print(f'Reshape X to : {X.shape}')
print('=================== y train ==================')
print(f'Shape of y : {y_train.shape}')
y_filled = filled_y(y_train, chunk_per_stim)
y_train = y_filled
print(f'Shape of new y : {y_train.shape}')

# 3.2 Train Test Val Split and Prepare X and y in correct shape
# 
# - For X, pytorch (if set batch_first) LSTM requires to be (batch, seq_len, features).  Thus, for us, it should be (100, 75, 16).
# - For y, nothing is special
# - So let's convert our numpy to pytorch, and then reshape using view

# 3.1 Chunking Val Set

chunk_size = 10

print('=================== X Val ==================')
print(f'Oringinal X shape {X_val.shape}')
X_val = chunk_data(X_val, chunk_size)
print(f'Chunked X : {X_val.shape}') # (#stim, #chunks, #electrodes, #datapoint per chunk)
chunk_per_stim = X_val.shape[1]
X_val = X_val.reshape(-1,len(electrodes),chunk_size)
print(f'Reshape X to : {X_val.shape}')
print('=================== y Val ==================')
print(f'Shape of y : {y_val.shape}')
y_filled = filled_y(y_val, chunk_per_stim)
y_val = y_filled
print(f'Shape of new y : {y_val.shape}')

# 3.1 Chunking Test Set

chunk_size = 10

print('=================== X Test ==================')
print(f'Oringinal X shape {X_test.shape}')
X_test = chunk_data(X_test, chunk_size)
print(f'Chunked X : {X_test.shape}') # (#stim, #chunks, #electrodes, #datapoint per chunk)
chunk_per_stim = X_test.shape[1]
X_test = X_test.reshape(-1,len(electrodes),chunk_size)
print(f'Reshape X to : {X_test.shape}')
print('=================== y Test ==================')
print(f'Shape of y : {y_test.shape}')
y_filled = filled_y(y_test, chunk_per_stim)
y_test = y_filled
print(f'Shape of new y : {y_test.shape}')

# 3.2.2 Convert to torch

torch_X_train = torch.from_numpy(X_train)
torch_y_train = torch.from_numpy(y_train)
check_torch_shape(torch_X_train,torch_X_train,'train')

torch_X_val = torch.from_numpy(X_val)
torch_y_val = torch.from_numpy(y_val)
check_torch_shape(torch_X_val,torch_y_val,'val')

torch_X_test = torch.from_numpy(X_test)
torch_y_test = torch.from_numpy(y_test)
check_torch_shape(torch_X_test,torch_y_test,'test')


# 3.2.3 Reshape

# CNN requires the input shape as (batch, channel, height, width)

torch_X_train_reshaped = torch_X_train.reshape(torch_X_train.shape[0],torch_X_train.shape[1],1,torch_X_train.shape[2])
print("Converted torch_X_train to ", torch_X_train_reshaped.size())

torch_X_val_reshaped = torch_X_val.reshape(torch_X_val.shape[0],torch_X_val.shape[1],1,torch_X_val.shape[2])
print("Converted torch_X_val to ", torch_X_val_reshaped.size())

torch_X_test_reshaped = torch_X_test.reshape(torch_X_test.shape[0],torch_X_test.shape[1],1,torch_X_test.shape[2])
print("Converted torch_X_test to ", torch_X_test_reshaped.size())


# 4. Dataset and DataLoader

from torch.utils.data import TensorDataset

BATCH_SIZE = 128 #keeping it binary so it fits GPU

#Train set loader
train_dataset = TensorDataset(torch_X_train_reshaped, torch_y_train)
train_iterator = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)
#Val set loader
val_dataset = TensorDataset(torch_X_val_reshaped, torch_y_val)
valid_iterator = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)
#Test set loader
test_dataset = TensorDataset(torch_X_test_reshaped, torch_y_test)
test_iterator = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE, 
                                           shuffle=True)


# 5. Training for Feature Extraction 

# 5.1 Define model parameters
# - Count model parameters
# - optimizer
# - loss function
# - GPU
model_EEGEncoder = EEGEncoder(input_size = len(electrodes))
model_EEGEncoder = model_EEGEncoder.float() #define precision as float to reduce running time
models = [model_EEGEncoder]

for model in models:
    print(f'The model {type(model).__name__} has {count_parameters(model):,} trainable parameters')# Train the model


# 5.2 Train the model
from train import train
from evaluate import evaluate

import torch.optim as optim

best_valid_loss = float('inf')
train_losses    = []
valid_losses    = []

learning_rate = 0.0001
N_EPOCHS      = 1500          ## best is 10k
criterion     = nn.CrossEntropyLoss()
optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate)


for model in models:
    model = model.to(device)
criterion = criterion.to(device)

model.is_debug = False
iteration = 0
classes = np.array(('Red', 'Green', 'Blue'))
for i, model in enumerate(models):
    print(f"Training {type(model).__name__}")

    start_time = time.time()

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc, train_predicted    = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc, valid_predicted, _ = evaluate(model, valid_iterator, criterion, classes, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        iteration     += 1

        if (epoch+1) % 50 == 0:
            clear_output(wait=True)
            print(f'Epoch: {epoch+1:02}/{N_EPOCHS}  |',end='')
            print(f'\tTrain Loss: {train_loss:.5f}  | Train Acc: {train_acc:.2f}%  |', end='')
            print(f'\t Val. Loss: {valid_loss:.5f}  | Val. Acc: {valid_acc:.2f}%')
            # do_plot(train_losses, valid_losses)


        if epoch > 500 and valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            #print("Model:{} saved.".format(type(model).__name__))
            try:
                os.makedirs('../model/03_FeatureExtraction/{par}/{roundno}/{electrode_zone}/{task}'.format(par=par,roundno=roundno,electrode_zone=electrode_zone,task=task))
            except:
                pass
            torch.save(model.state_dict(), "../model/03_FeatureExtraction/{par}/{roundno}/{electrode_zone}/{task}/EEG_ENCODER_{fmin}_{fmax}_{tmin}_{tmax}.pt.tar".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax))
            best_model_index = i
            best_train_acc = train_acc
            best_val_acc = valid_acc


# 6. Evaluation [Test set]
# Define classes

classes = np.array(('Red', 'Green', 'Blue'))
model = EEGEncoder(input_size = len(electrodes))
model = model.float()
model = model.to(device)
model.load_state_dict(torch.load("../model/03_FeatureExtraction/{par}/{roundno}/{electrode_zone}/{task}/EEG_ENCODER_{fmin}_{fmax}_{tmin}_{tmax}.pt.tar".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax)))

test_loss, test_acc , predicted, actual_labels, acc_class_test = evaluate(model, test_iterator, criterion, classes, device, test = True)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')
print("---------------")
print(" (Actual y , Predicted y)")

y_test_t = squeeze_to_list(actual_labels)
y_hat_test_t = squeeze_to_list(predicted)

out_test = zip(y_test_t, y_hat_test_t)

# 7. Evaluation [Real Test]
# X_real_test = chunk_data(X_real_test, chunk_size)
# chunk_per_stim = X_real_test.shape[1]
# X_real_test = X_real_test.reshape(-1,len(electrodes),chunk_size)
# y_filled_real_test = filled_y(y_real_test, chunk_per_stim)

# print("Chucked X_test: ",X_real_test.shape )
# print("y_filled_test: ",y_filled_real_test.shape )

# # torch_X_real_test = torch.from_numpy(X_real_test)
# torch_y_real_test = torch.from_numpy(y_filled_real_test)
# check_torch_shape(torch_X_real_test,torch_y_real_test,'test')

# print("Shape of torch_X: ",torch_X_real_test.shape)
# print("Shape of torch_y: ",torch_y_real_test.shape)

# torch_X_real_test_reshaped = torch_X_real_test.reshape(torch_X_real_test.shape[0],torch_X_real_test.shape[1],1,torch_X_real_test.shape[2])
# print("Converted X to ", torch_X_real_test_reshaped.size())

# real_test_dataset = TensorDataset(torch_X_real_test_reshaped, torch_y_real_test)
#Test set loader
# real_test_iterator = torch.utils.data.DataLoader(dataset=real_test_dataset, 
#                                           batch_size=BATCH_SIZE, 
#                                           shuffle=True)

# model = EEGEncoder(input_size = len(electrodes))
# model = model.float()
# model = model.to(device)
# model.load_state_dict(torch.load("../model/03_FeatureExtraction/{par}/{roundno}/{electrode_zone}/{task}/EEG_ENCODER_{fmin}_{fmax}.pt.tar".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax)))

# test_loss, real_test_acc , predicted, actual_labels, acc_class_real_test = evaluate(model, real_test_iterator, criterion, classes, device, test=True)
# print(f'Test Loss: {test_loss:.3f} | Test Acc: {real_test_acc:.2f}%')
# print("---------------")
# print(" (Actual y , Predicted y)")

# y_test_rt    = squeeze_to_list(actual_labels)
# y_hat_test_rt = squeeze_to_list(predicted)

# out_real_test = zip(y_test_rt, y_hat_test_rt)
# print(list(out_real_test))

# 8. Extracted Features
X_train_val = np.concatenate((X_train,X_val))
y_train_val = np.concatenate((y_train,y_val))

print(X_train_val.shape)
print(y_train_val.shape)

torch_X_train_val = torch.from_numpy(X_train_val)
torch_y_train_val = torch.from_numpy(y_train_val)
check_torch_shape(torch_X_train_val,torch_y_train_val,'train_val')

torch_X_train_val_reshaped = torch_X_train_val.reshape(torch_X_train_val.shape[0],torch_X_train_val.shape[1],1,torch_X_train_val.shape[2])
print("Converted X to ", torch_X_train_val_reshaped.size())

# save extracted features
eeg_encode = model.get_latent(torch_X_train_val_reshaped.to(device).float())
eeg_extracted_features = eeg_encode.detach().cpu().numpy()


# 9. SAVE
try:
    os.makedirs('../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}'.format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone))
except:
    pass
# Save Real Test
# np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/X_real_test_{fmin}_{fmax}".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax),X_real_test)
# np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/y_real_test_{fmin}_{fmax}".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax),y_filled_real_test)

# Save Train
if psd_enable:
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/X_train_{fmin}_{fmax}_{tmin}_{tmax}_psd".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),X_train)
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/y_train_{fmin}_{fmax}_{tmin}_{tmax}_psd".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),y_train)

    # Save Test
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/X_test_{fmin}_{fmax}_{tmin}_{tmax}_psd".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),X_test)
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/y_test_{fmin}_{fmax}_{tmin}_{tmax}_psd".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),y_test)

    # Save Val
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/X_val_{fmin}_{fmax}_{tmin}_{tmax}_psd".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),X_val)
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/y_val_{fmin}_{fmax}_{tmin}_{tmax}_psd".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),y_val)

    # Save Extracted Features
    np.save('../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/extracted_features_X_{fmin}_{fmax}_{tmin}_{tmax}_psd'.format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax), eeg_extracted_features )
    np.save('../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/extracted_features_y_{fmin}_{fmax}_{tmin}_{tmax}_psd'.format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax), y_train_val)

else:
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/X_train_{fmin}_{fmax}_{tmin}_{tmax}".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),X_train)
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/y_train_{fmin}_{fmax}_{tmin}_{tmax}".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),y_train)

    # Save Test
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/X_test_{fmin}_{fmax}_{tmin}_{tmax}".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),X_test)
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/y_test_{fmin}_{fmax}_{tmin}_{tmax}".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),y_test)

    # Save Val
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/X_val_{fmin}_{fmax}_{tmin}_{tmax}".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),X_val)
    np.save("../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/y_val_{fmin}_{fmax}_{tmin}_{tmax}".format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax),y_val)

    # Save Extracted Features
    np.save('../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/extracted_features_X_{fmin}_{fmax}_{tmin}_{tmax}'.format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax), eeg_extracted_features )
    np.save('../data/participants/{par}/03_FeatureExtraction/{roundno}/{electrode_zone}/{task}/extracted_features_y_{fmin}_{fmax}_{tmin}_{tmax}'.format(par=par,task=task,roundno=roundno,electrode_zone=electrode_zone,fmin=fmin,fmax=fmax, tmin=tmin, tmax=tmax), y_train_val)


# 10. Results
try:
    os.makedirs('../results')
except:
    pass
# with open(f"../results/classification_results_{task}.txt", "a") as myfile:
#     myfile.write(f'================= {par}:round{roundno}:{fmin}-{fmax}:{tmin}-{tmax} ================\n')
#     myfile.write(f" Train Acc: {train_acc} \n Valid Acc: {valid_acc} \n Test Acc: {test_acc} \n") # Real test Acc: {real_test_acc} \n")
#     myfile.write("------- Acc per class for test ------- \n")
#     for v,k in acc_class_test.items():
#         myfile.write(f"{v}: {k[0]} \n")
    # myfile.write("---- Acc per class for real test ----- \n")
    # for v,k in acc_class_real_test.items():
    #     myfile.write(f"{v}: {k[0]} \n")

RGB = list(acc_class_test.keys())
filename = "../results/classification_results.csv"
mode = 'a' if os.path.exists(filename) else 'w'
with open(f"../results/classification_results.csv", mode) as myfile:
    fileEmpty = os.stat(filename).st_size == 0
    headers = ['Participant','Round' , 'Task', 'PSD','Electrode Zone', 'fmin', 'fmax', 'tmin', 'tmax', 'Train Acc', 'Val Acc', 'Test Acc', 'Red Acc', 'Green Acc', 'Blue Acc']
    writer = csv.DictWriter(myfile, delimiter=',', lineterminator='\n',fieldnames=headers)
    if fileEmpty:
        writer.writeheader()  # file doesn't exist yet, write a header
    writer.writerow( {'Participant': par, 'Round': roundno, 'Task': task, 'PSD': psd_enable, 'Electrode Zone': electrode_zone, 'fmin': fmin, 'fmax': fmax, 'tmin': tmin, 'tmax': tmax, 'Train Acc': best_train_acc, 'Val Acc': best_val_acc, 'Test Acc': test_acc, 'Red Acc': acc_class_test[RGB[0]][0], 'Green Acc': acc_class_test[RGB[1]][0], 'Blue Acc': acc_class_test[RGB[2]][0]}) #{real_test_acc},")
