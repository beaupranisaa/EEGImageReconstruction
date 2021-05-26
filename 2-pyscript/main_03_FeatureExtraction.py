import os 
import time
from config import *

print('Total loop:', len(files)*round*len(electrode_zones)*len(tasks)*len(fmax_range))
time.sleep(5)
for f in files:
    for r in range(round):
        for electrode_zone in electrode_zones:
            electrodes = electrode_zones[electrode_zone]
            for task in tasks:
                for j in range(len(fmax_range)):
                    i = int(f.split("-")[0])
                    os.system(f'python3 03_FeatureExtraction/03_FeatureExtraction.py par{i} {f} {fmin_range[j]} {fmax_range[j]} {task} {electrode_zone} {electrodes} {r+1}')