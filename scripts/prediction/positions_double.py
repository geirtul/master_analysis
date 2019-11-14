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

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Import Data ==================
images = np.load(DATA_PATH + "images_1M.npy")
positions = np.load(DATA_PATH + "positions_1M.npy")
#labels = np.load(DATA_PATH + "labels_noscale_200k.npy")

# ================== Prepare Data ==================

# Set positions x2,y2 for single events to -1 instead of -100
# Plan is to predict two positions regardless of single or double
# event, but for single events x2,y2 should be predicted out of bounds.
single_indices, double_indices, close_indices = event_indices(positions)

# Split indices into training and test sets
x_idx = np.arange(images.shape[0])
train_idx, test_idx, not_used1, not_used2 = train_test_split(
        x_idx, 
        x_idx, 
        test_size = 0.2
        )  
single_train, double_train, dummy2 = event_indices(positions[train_idx])
single_test, double_test, dummy2 = event_indices(positions[test_idx])

positions[single_indices, 2:] = -1.0
positions[single_indices, :2] *= 3 # Scale to mm instead of pixels
positions[double_indices] *= 3
# ================== Custom Functions ==================
# Define R2 score for metrics since it's not available by default
def r2_keras(y_true, y_pred):
    SS_res =  backend.sum(backend.square(y_true - y_pred)) 
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + backend.epsilon()) )


# ================== Model ==================
modeltype = "cnn"
with tf.device('/GPU:2'):
    if modeltype == "dense":
        model = position_dense()
        images = images.reshape((images.shape[0], 256))
    elif modeltype == "project":
        model = position_project()
    else:
        model = position_cnn()


    # Setup callback for saving models
    fpath = MODEL_PATH + modeltype + "-r2-{val_r2_keras:.2f}.hdf5"
    cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=fpath, 
            monitor='val_r2_keras', 
            save_best_only=True,
            mode='max'
            )

    # Compile model
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=[r2_keras])

    # Parameters for the model
    batch_size = 32
    epochs = 10

    history = model.fit(
            images[train_idx][double_train],
            positions[train_idx][double_train],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(images[test_idx][double_test], positions[test_idx][double_test]),
            callbacks=[cb]
            )

