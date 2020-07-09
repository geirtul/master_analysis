from master_scripts.data_functions import import_data
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# Load data from file and save static training and validation set
# for initial hyperparameter searches

DATA_PATH = "../data/simulated/"
fname = "CeBr200k_Mix.txt"
images, energies, positions, labels = import_data(DATA_PATH + fname)
x_idx = np.arange(images.shape[0])
train_idx, val_idx, unused1, unused2 = train_test_split(x_idx, x_idx)
np.save(DATA_PATH + "images_200k_train.npy", images[train_idx])
np.save(DATA_PATH + "energies_200k_train.npy", energies[train_idx])
np.save(DATA_PATH + "positions_200k_train.npy", positions[train_idx])
np.save(DATA_PATH + "labels_200k_train.npy", labels[train_idx])

np.save(DATA_PATH + "images_200k_val.npy", images[val_idx])
np.save(DATA_PATH + "energies_200k_val.npy", energies[val_idx])
np.save(DATA_PATH + "positions_200k_val.npy", positions[val_idx])
np.save(DATA_PATH + "labels_200k_val.npy", labels[val_idx])
