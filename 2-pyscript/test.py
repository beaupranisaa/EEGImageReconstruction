from ast import fix_missing_locations
from mne.preprocessing.ica import _find_max_corrs
import numpy as np
from scipy.stats.stats import tmax

par = "par1"
file = "1-par1"
task = "perception"
fmin = float("0.01")
fmax = float("4.0")
tmin = float("0.0")
tmax = float("0.5")
electrode_zone = "all"
psd_enable = True

# /root/HCI/EEGImageReconstruction/data/participants/par1/02_ArtifactRemoval_Epoching/1-par1_perception_0.01_4.0_0.0_0.5_all_psd_X.npy
                                # ../data/participants/par1/02_ArtifactRemoval_Epoching/1-par_perception_0.01_4.0_0.0_0.5_all_psd_X.npy    
if psd_enable is True:
    # path ='../data/participants/par1/02_ArtifactRemoval_Epoching/1-par1_perception_0.01_4.0_0.0_0.5_all_psd_X.npy'
    path ='../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_psd_X.npy'.format(par=par,file=file, task=task,fmin = fmin, fmax = fmax, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone)
    print(path)
    X_ = np.load(path, allow_pickle=True)
    print(X_[0])
    #y = np.load('../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_psd_y.npy'.format(par=par,file=file, task=task,fmin = fmin, fmax = fmax, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), allow_pickle=True)

else:
    X_ = np.load('../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_X.npy'.format(par=par,file=file, task=task,fmin = fmin, fmax = fmax, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), allow_pickle=True)
    y = np.load('../data/participants/{par}/02_ArtifactRemoval_Epoching/{file}_{task}_{fmin}_{fmax}_{tmin}_{tmax}_{electrode_zone}_y.npy'.format(par=par,file=file, task=task,fmin = fmin, fmax = fmax, tmin=tmin, tmax=tmax, electrode_zone=electrode_zone), allow_pickle=True)
    print(X_[0])