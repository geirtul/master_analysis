# Imports
import numpy as np
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Import Data ==================
images = np.load(DATA_PATH + "images_noscale_200k.npy")
positions = np.load(DATA_PATH + "positions_noscale_200k.npy")
energies = np.load(DATA_PATH + "energies_noscale_200k.npy")
#images = normalize_image_data(images)
#images = np.load(DATA_PATH + "images_1M.npy")
#positions = np.load(DATA_PATH + "positions_1M.npy")
#labels = np.load(DATA_PATH + "labels_noscale_200k.npy")

# ================== Prepare Data ==================
images = images.reshape(images.shape[0], 256)
data = np.concatenate((images, positions, energies), axis=1)

corr_matrix = np.corrcoef(data, rowvar=False)
plt.imshow(corr_matrix)
plt.colorbar()
plt.savefig("corr_matrix.pdf")
plt.clf()

# Correlations for positions and energy, double events
single_idx, double_idx, close_idx = event_indices(positions)
corr_matrix_pos_energy = np.corrcoef(data[double_idx, 256:], rowvar=False)

plt.imshow(corr_matrix_pos_energy)
plt.colorbar()
plt.savefig("corr_matrix_pos_energy.pdf")
plt.clf()


