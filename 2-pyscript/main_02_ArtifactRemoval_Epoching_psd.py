import os 
from config import *

for f in files:
    i = int(f.split("-")[0])
    for electrode_zone in electrode_zones:
        electrodes = electrode_zones[electrode_zone]
        for j in range(len(tmax_range)):
            for k in range(len(fmax_range)):
                print(f'Participant: {i}, Electrode Zone: {electrode_zone}, Frequency: {fmin_range[k]}-{fmax_range[k]}, Time: {tmin_range[j]}-{tmax_range[j]}, PSD: {psd}')
                os.system(f'python3 02_ArtifactRemoval_Epoching_psd/02_ArtifactRemoval_Epoching_psd.py par{i} {f} {fmin_range[k]} {fmax_range[k]} {tmin_range[j]} {tmax_range[j]} {electrode_zone} {electrodes} {psd}')
        #         break
        #     break
        # break
    break 