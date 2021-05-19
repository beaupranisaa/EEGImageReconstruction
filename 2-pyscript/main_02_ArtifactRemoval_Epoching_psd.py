import os 
from config import *

for f in files:
    i = int(f.split("-")[0])
    os.system(f'python3 02_ArtifactRemoval_Epoching_psd/02_ArtifactRemoval_Epoching_psd.py par{i} {f} {fmin} {fmax}')
