# Imports
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
import re
import subprocess
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from analysis_functions.callbacks import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import backend
from master_models.classification import project_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, StratifiedKFold

# Determine tensorflow device
MAX_LOAD = 20 # maximum GPU-util 20%
DEVICE = get_tf_device(MAX_LOAD)  

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Import Data ==================
images = np.load(DATA_PATH + "images_full.npy")
positions = np.load(DATA_PATH + "positions_full.npy")
labels = np.load(DATA_PATH + "labels_full.npy")
labels = to_categorical(labels)
#images = normalize_image_data(images)
#images = np.load(DATA_PATH + "images_1M.npy")
#positions = np.load(DATA_PATH + "positions_1M.npy")
#labels = np.load(DATA_PATH + "labels_noscale_200k.npy")
# ================== Prepare Data ==================

single_indices, double_indices, close_indices = event_indices(positions)
positions = normalize_position_data(positions)

# Split indices into training, validation, and test sets
# The test set is the "out-of-sample" set, used only for the
# "final score" after training
x_idx = np.arange(images.shape[0])
trainval_idx, test_idx, not_used1, not_used2 = train_test_split(
        x_idx, 
        x_idx, 
        test_size = 0.2
        ) 

train_idx, val_idx, not_used3, not_used4 = train_test_split(
        trainval_idx, 
        trainval_idx, 
        test_size = 0.2
        ) 
with tf.device(DEVICE):
    model = project_model()

    # Callbacks to save best model
    # Setup callback for saving models
    fpath = MODEL_PATH + "project" + "-{val_acc:.2f}.hdf5"
    cb_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=fpath, 
            monitor='val_acc', 
            save_best_only=True
            )

    cb_earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_acc', 
            patience=2,
            verbose=1,
            restore_best_weights=True,
            )
    # Params for k-fold cross-validation
    #k_splits = 5
    #k_shuffle = True
    epochs = 10
    batch_size = 32
    #kfold_labels = labels.argmax(axis=-1) # StratifiedKFold doesn't take one-hot
    # Define device scope and fit the data
    model.fit(
            normalize_image_data(images[train_idx]), 
            labels[train_idx],
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(normalize_image_data(images[val_idx]), labels[val_idx]),
            callbacks=[cb_save, cb_earlystopping],
            )
    model_eval = model.evaluate(
            normalize_image_data(images[test_idx]), 
            labels[test_idx]
            )
    print('test loss, test acc:', model_eval)




