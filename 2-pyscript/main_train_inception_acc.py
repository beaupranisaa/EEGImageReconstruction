import os 
from config import *

for f in files:
    i = int(f.split("-")[0])
    os.system(f'python3 04_Image_Reconstruction/train_inception_acc.py par{i} {f} {task}')