import os 
from config import *

for f in files:
    i = int(f.split("-")[0])
    os.system(f"python3 01_Data_excel-pd/01_Data_excel-pd.py par{i} {f}")
