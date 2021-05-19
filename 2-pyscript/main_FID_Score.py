import os 
from config import *

for f in files:
    i = int(f.split("-")[0])
    os.system(f'python3 04_Image_Reconstruction/FID_Score.py par{i} {task} {test_type}')

