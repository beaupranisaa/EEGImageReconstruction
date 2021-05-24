import numpy as np

# perception or imagery
# relevant for 03_FeatureExtraction.py and 04_Image_Reconstruction.py and FID_score.py
task = "perception"

# frequency range
# relevant for 02_ArtifactRemoval_Epoching_psd.py and 03_FeatureExtraction.py
fmin = 0.01
fmax = 1 

# real_test or test
# relevant for FID_score.py
test_type = "test"

files = [
"1-par1",
"2-par2",
"3-par3",
"4-par4",
"5-par5",
"6-par6",
"7-par7",
"8-par8",
"9-par9",
"10-par10",
"11-par11",
"12-par12",
"13-par13",
"14-par14",
"15-par15",
"16-par16",
"17-par17",
"18-par18",
"19-par19",
"20-par20",
]

# electrodes
# relevant for 03_FeatureExtraction.py
# F3 F8 F7 F4 P3 P4 T4 T3 Fp2 Fp1 C4 C3 T6 T5 O1 O2
# 0  1  2  3  4  5  6  7  8   9   10 11 12 13 14 15
electrodes_F = "[0,1,2,3]"
electrodes_P = "[4,5]"
electrodes_T = "[6,7,12,13]"
electrodes_Fp = "[8,9]"
electrodes_C = "[10,11]"
electrodes_O = "[14,15]"
electrodes_all = "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]"

electrodes = electrodes_all
