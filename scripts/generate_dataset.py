import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from master_data_functions.functions import import_data, event_indices


# Generate a dataset of desired number of samples

DATA_PATH = "../data/simulated/"
fname = "CeBr2Mil_Mix.txt"
data = import_data(DATA_PATH+fname, scaling=False)
print("Data imported")

singles, doubles, close = event_indices(data["positions"][:1900000])
singles2, doubles2, close2 = event_indices(data["positions"][1900000:])

print("Num singles:", len(singles))
print("Num doubles:", len(doubles))
print("Num close:", len(close))
print("Test set")
print("Num singles:", len(singles2))
print("Num doubles:", len(doubles2))
print("Num close:", len(close2))
print("Saving energies")

np.save(DATA_PATH+"energies_full.npy", data["energies"][:1900000])
print("Saving positions")
np.save(DATA_PATH+"positions_full.npy", data["positions"][:1900000])
print("Saving images")
np.save(DATA_PATH+"images_full.npy", data["images"][:1900000])
print("Saving labels")
np.save(DATA_PATH+"labels_full.npy", data["labels"][:1900000])
print("Saving test set ================")
np.save(DATA_PATH+"energies_test.npy", data["energies"][1900000:])
print("Saving positions")
np.save(DATA_PATH+"positions_test.npy", data["positions"][1900000:])
print("Saving images")
np.save(DATA_PATH+"images_test.npy", data["images"][1900000:])
print("Saving labels")
np.save(DATA_PATH+"labels_test.npy", data["labels"][1900000:])
