# Imports
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from master_models.classification import labollita_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, StratifiedKFold

# Path to data on ML-server
DATA_PATH_ML = "../../data/simulated/"
images = load_feature_representation(filename="images_noscale_200k.npy", path=DATA_PATH_ML)
energies = load_feature_representation(filename="energies_noscale_200k.npy", path=DATA_PATH_ML)
positions = load_feature_representation(filename="positions_noscale_200k.npy", path=DATA_PATH_ML)
labels = load_feature_representation(filename="labels_noscale_200k.npy", path=DATA_PATH_ML)

# Normalize image data
images = normalize_image_data(images)

n_classes = len(np.unique(labels))

# Indices to use for training and test
x_idx = np.arange(images.shape[0])

# Split the indices into training and test sets
train_idx, test_idx, not_used1, not_used2 = train_test_split(x_idx, x_idx, test_size = 0.2)    

# x_train = images[train_idx]
# x_test = images[test_idx]
# y_test = labels[test_idx]
# y_train = labels[train_idx]

# Initialize model and train
model = labollita_model()

# Callbacks to save best model
# Setup callback for saving models
fpath = OUTPUT_PATH + net + "-{val_accuracy:.2f}.hdf5"
cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=fpath, 
        monitor='val_accuracy', 
        save_best_only=True
        )

# Define device scope and fit the data
with tf.device('/device:GPU:0'):
    model.fit(images[train_idx], labels[train_idx])




