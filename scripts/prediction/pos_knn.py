# Imports
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from master_models.prediction import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend
from sklearn.neighbors import KNeighborsRegressor

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Import Data ==================
#images = np.load(DATA_PATH + "images_noscale_200k.npy")
#positions = np.load(DATA_PATH + "positions_noscale_200k.npy")
#images = normalize_image_data(images)
images = np.load(DATA_PATH + "images_1M.npy")
positions = np.load(DATA_PATH + "positions_1M.npy")
energies = np.load(DATA_PATH + "energies_1M.npy")
#labels = np.load(DATA_PATH + "labels_noscale_200k.npy")
# ================== Prepare Data ==================

# Set positions x2,y2 for single events to -1 instead of -100
# Plan is to predict two positions regardless of single or double
# event, but for single events x2,y2 should be predicted out of bounds.
single_indices, double_indices, close_indices = event_indices(positions)
positions[single_indices, 2:] = -1.0
positions[single_indices, :2] /= 16 # Scale to mm instead of pixels
positions[double_indices] /= 16

# Split indices into training and test sets
x_idx = np.arange(images.shape[0])
train_idx, test_idx, not_used1, not_used2 = train_test_split(
        x_idx, 
        x_idx, 
        test_size = 0.2
        ) 
images = images.reshape(images.shape[0], 256)

# Save training and test indices, and also the test set
#np.save("train_idx.npy", train_idx)
#np.save("test_idx.npy", test_idx)
#np.save("test_images__double_1M.npy", images[test_idx])
#np.save("test_positions_double_1M.npy", positions[test_idx])
#np.save("test_energies_double_1M.npy", energies[test_idx])
# ================== Custom Functions ==================
# Define R2 score for metrics since it's not available by default
def r2_keras(y_true, y_pred):
    SS_res =  backend.sum(backend.square(y_true - y_pred)) 
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + backend.epsilon()) )

# ================== Model ==================
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(images[train_idx], positions[train_idx])
print(neigh.score(images[test_idx], positions[test_idx]))


