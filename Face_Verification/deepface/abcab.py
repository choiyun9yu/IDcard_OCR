from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]                       
threshold = 0
try: 
    with open(f"{ROOT}/rep/Train/Threshold.txt", "r") as f:
        threshold = f.readline()
    pass
except:
    pass