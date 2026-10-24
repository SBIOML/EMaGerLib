import numpy as np
from pathlib import Path

file = Path(r"C:\Users\felix\OneDrive\Documents\phD\Projet\CODING\EMaGerLib\Datasets\test_emager_v11\C_5_R_7_emg.csv")

data = np.loadtxt(file, delimiter=",")

print("File:", file.name)
print("Shape:", data.shape)

if data.shape[0] == 64:
    print("Format seems to be: channels x samples")
    print("Estimated duration:", data.shape[1] / 1000, "seconds")

elif data.shape[1] == 64:
    print("Format seems to be: samples x channels")
    print("Estimated duration:", data.shape[0] / 1000, "seconds")

else:
    print("Unexpected shape. Not sure which axis is time.")