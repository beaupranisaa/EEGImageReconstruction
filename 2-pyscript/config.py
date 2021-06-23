from time import time
import numpy as np

# perception or imagery
# relevant for 03_FeatureExtraction.py and 04_Image_Reconstruction.py and FID_score.py
tasks = ["perception", "imagery"]

# frequency range
# relevant for 02_ArtifactRemoval_Epoching_psd.py and 03_FeatureExtraction.py
#fmin = 0.01
#fmax = 1 
start_freq = 1
stop_freq = 41
window_freq = 1
overlap_freq = 3
fmin_s = np.array(0.01)
fmin = np.arange(start_freq,stop_freq - overlap_freq, window_freq)
fmin_range = np.append(fmin_s,fmin)
fmax_range = np.arange(start_freq + overlap_freq, stop_freq, window_freq)

# time range
# relevant for 02_ArtifactRemoval_Epoching_psd.py and 03_FeatureExtraction.py
start_time = 0
stop_time = 2000
window_time = 500
overlap_time = 250
tmin_range = np.arange(start_time, stop_time - overlap_time, window_time)
tmax_range = np.arange(start_time + overlap_time, stop_time, window_time)

# real_test or test
# relevant for FID_score.py
test_type = "test"

files = [
"1-par1",
# "2-par2",
# "3-par3",
# "4-par4",
# "5-par5",
# "6-par6",
# "7-par7",
# "8-par8",
# "9-par9",
# "10-par10",
# "11-par11",
# "12-par12",
# "13-par13",
# "14-par14",
# "15-par15",
# "16-par16",
# "17-par17",
# "18-par18",
# "19-par19",
# "20-par20",
]

# electrodes
# relevant for 03_FeatureExtraction.py
# F3 F8 F7 F4 P3 P4 T4 T3 Fp2 Fp1 C4 C3 T6 T5 O1 O2
# 0  1  2  3  4  5  6  7  8   9   10 11 12 13 14 15

# electrode_zones = {
# # "F" : "[0,1,2,3]",
# # "P" : "[4,5]",
# # "T" : "[6,7,12,13]",
# # "Fp" : "[8,9]",
# # "C" : "[10,11]",
# # "O" : "[14,15]",
# "all" : "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]",
# }

electrode_zones = {
# "F" : "[0,1,2,3]",
# "P" : "[4,5]",
# "T" : "[6,7,12,13]",
# "Fp" : "[8,9]",
# "C" : "[10,11]",
# "O" : "[14,15]",
"all" : "[F3,F8,F7,F4,P3,P4,T4,T3,Fp2,Fp1,C4,C3,T6,T5,O1,O2]",
}



# electrode_zone = "all"
# electrodes = electrode_zones[electrode_zone]

round = 5