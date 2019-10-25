# Imports
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, StratifiedKFold



# Load existing data. The feature_rep functions just use numpy storage.
# Image are allready reshaped to (16, 16, 3) here
images = load_feature_representation(filename="images_noscale_200k.npy")
energies = load_feature_representation(filename="energies_noscale_200k.npy")
positions = load_feature_representation(filename="positions_noscale_200k.npy")
labels = load_feature_representation(filename="labels_noscale_200k.npy")

images = normalize_image_data(images)
images = np.concatenate((images, images, images), axis=3)
n_classes = len(np.unique(labels))


