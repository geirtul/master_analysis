# Imports
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from master_models.classification import project_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, StratifiedKFold

with tf.device('/job:localhost/replica:0/task:0/device:GPU:3'):
    # Path to data on ML-server
    DATA_PATH = "../../data/simulated/"
    OUTPUT_PATH = "../../data/output/"
    MODEL_OUTPUT_PATH = OUTPUT_PATH + "models/"

    images = np.load(DATA_PATH + "images_noscale_200k.npy")
    energies = np.load(DATA_PATH + "energies_noscale_200k.npy")
    positions = np.load(DATA_PATH + "positions_noscale_200k.npy")
    labels = np.load(DATA_PATH + "labels_noscale_200k.npy")

    # Normalize image data
    images = normalize_image_data(images)

    n_classes = len(np.unique(labels))

    # Indices to use for training and test
    x_idx = np.arange(images.shape[0])

    # Split the indices into training and test sets
    train_idx, test_idx, not_used1, not_used2 = train_test_split(x_idx, x_idx, test_size = 0.2)    

    # Initialize model and train
    model = project_model()

    # Callbacks to save best model
    # Setup callback for saving models
    fpath = MODEL_OUTPUT_PATH + "project" + "-{val_acc:.2f}.hdf5"
    cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=fpath, 
            monitor='val_acc', 
            save_best_only=True
            )

    # Params for k-fold cross-validation
    #k_splits = 5
    #k_shuffle = True
    epochs = 5
    batch_size = 32
    #kfold_labels = labels.argmax(axis=-1) # StratifiedKFold doesn't take one-hot
    # Define device scope and fit the data
    model.fit(
            images[train_idx], 
            labels[train_idx],
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(images[test_idx], labels[test_idx]),
            callbacks=[cb]
            )




