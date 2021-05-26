import os 
from config import *

for f in files:
    for r in range(round):
        i = int(f.split("-")[0])
        os.system(f'python3 03_FeatureExtraction/03_FeatureExtraction.py par{i} {f} {fmin} {fmax} {task} {electrode_zone} {electrodes} {r+1}')
        break