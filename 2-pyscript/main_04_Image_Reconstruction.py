import os 
from config import *

for f in files:
    i = int(f.split("-")[0])
    os.system(f'python3 04_Image_Reconstruction/04_Image_Reconstruction.py par{i} {f} {task}')


