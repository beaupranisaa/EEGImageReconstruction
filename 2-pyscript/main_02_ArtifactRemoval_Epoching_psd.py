import os 
from config import *

for f in files:
    i = int(f.split("-")[0])
    for j in range(len(fmax_range)):
        os.system(f'python3 02_ArtifactRemoval_Epoching_psd/02_ArtifactRemoval_Epoching_psd.py par{i} {f} {fmin_range[j]} {fmax_range[j]}')
