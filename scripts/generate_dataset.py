import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from master_data_functions.functions import import_data, event_indices


# Generate a dataset of desired number of samples

DATA_PATH = "../data/sample/"
fname = "CeBr10k_1.txt"
data = import_data(DATA_PATH+fname, scaling=False)
print("Data imported")

singles, doubles, close = event_indices(data["positions"])

np.save(DATA_PATH+"energies_sample.npy", data["energies"])
print("Saving positions")
np.save(DATA_PATH+"positions_sample.npy", data["positions"])
print("Saving images")
np.save(DATA_PATH+"images_sample.npy", data["images"])
print("Saving labels")
np.save(DATA_PATH+"labels_sample.npy", data["labels"])
