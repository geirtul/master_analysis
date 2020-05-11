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
from master_models.prediction import *
from analysis_functions.callbacks import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import backend

# Determine tensorflow device
MAX_LOAD = 20 # maximum GPU-util 20%
DEVICE = get_tf_device(MAX_LOAD)  

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Import Data ==================
images = np.load(DATA_PATH + "images_noscale_200k.npy")
positions = np.load(DATA_PATH + "positions_noscale_200k.npy")
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
        single_indices, 
        single_indices, 
        test_size = 0.2
        ) 

train_idx, val_idx, not_used3, not_used4 = train_test_split(
        trainval_idx, 
        trainval_idx, 
        test_size = 0.2
        ) 

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
with tf.device(DEVICE):
    model = position_single_cnn()
    val_data = (normalize_image_data(images[val_idx]), positions[val_idx,:2])
    # Setup callback for saving models
    fpath = MODEL_PATH + "cnn_pos_single_" + "mse_{val_loss:.5f}.hdf5"
    cb_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=fpath, 
            monitor='val_loss', 
            save_best_only=True,
            mode='min'
            )
    cb_earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3,
            )

    cb_r2 = R2ScoreCallback(val_data)

    # Compile model
    ## Custom optimizer
    #curr_adam = tf.keras.optimizers.Adam(lr=lmbda)
    model.compile(loss='mse',
                  optimizer='adam')
    print(model.summary())

    # Parameters for the model
    batch_size = 32
    epochs = 10
    history = model.fit(
            normalize_image_data(images[train_idx]),
            positions[train_idx,:2],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_data,
            verbose=2,
            callbacks=[cb_earlystopping, cb_save, cb_r2]
            )

