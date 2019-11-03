# Imports
import numpy as np
import tensorflow-gpu as tf
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from master_models.classification import labollita_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, StratifiedKFold

# tensorflow configuration
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



exit(1)
# Load existing data. The feature_rep functions just use numpy storage.
# Image are allready reshaped to (16, 16, 3) here
images = load_feature_representation(filename="images_noscale_200k.npy")
energies = load_feature_representation(filename="energies_noscale_200k.npy")
positions = load_feature_representation(filename="positions_noscale_200k.npy")
labels = load_feature_representation(filename="labels_noscale_200k.npy")

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
model.train(images[train_idx], labels[train_idx])




