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
np.save(DATA_PATH + "images_200k.npy", images)
np.save(DATA_PATH + "energies_200k.npy", energies)
np.save(DATA_PATH + "positions_200k.npy", positions)
np.save(DATA_PATH + "labels_200k.npy", labels)
