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
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras import backend

# Determine tensorflow device
# Checks for GPU availability and sets DEVICE
DEVICE = None
MAX_LOAD = 20 # maximum GPU-util 20%
# Hacky but works for checking if version is < 2 for ML-servers
if int(tf.__version__[0]) < 2:
    gpu_devices = tf.config.experimental.list_logical_devices('GPU')
    cpu_devices = tf.config.experimental.list_logical_devices('CPU')
else:
    gpu_devices = tf.config.list_logical_devices('GPU')
    cpu_devices = tf.config.list_logical_devices('CPU')


if gpu_devices:
    nvidia_command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv"]
    nvidia_output = subprocess.run(nvidia_command, text=True, capture_output=True).stdout
    gpu_loads = np.array(re.findall(r"(\d+), (\d+) %", nvidia_output),
    dtype=np.int) # tuple (id, load%)
    eligible_gpu = np.where(gpu_loads[:,1] < MAX_LOAD)
    if len(eligible_gpu[0]) == 0:
        print("No GPUs with less than 20% load. Check nvidia-smi.")
        exit(0)
    else:
        # Choose the highest id eligible GPU
        # Assuming a lot of people use default allocation which is
        # lowest id.
        gpu_id = np.amax(gpu_loads[eligible_gpu,0])
        DEVICE = gpu_devices[gpu_id].name
        print("CHOSEN GPU IS:", DEVICE)
else:
    # Default to CPU
    DEVICE = cpu_devices[0].name
    print("NO GPU FOUND, DEFAULTING TO CPU.")

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
#labels = np.load(DATA_PATH + "labels_noscale_200k.npy")
# ================== Prepare Data ==================

single_indices, double_indices, close_indices = event_indices(positions)
positions = normalize_position_data(positions)

# Split indices into training and test sets
x_idx = np.arange(images.shape[0])
train_idx, test_idx, not_used1, not_used2 = train_test_split(
        single_indices, 
        single_indices, 
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
    # Setup callback for saving models
    fpath = MODEL_PATH + "cnn_pos_single_" + "r2_{val_r2_keras:.2f}.hdf5"
    cb_save = tf.keras.callbacks.ModelCheckpoint(
            filepath=fpath, 
            monitor='val_r2_keras', 
            save_best_only=True,
            mode='max'
            )
    cb_earlystopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3,
            )


    # Compile model
    ## Custom optimizer
    #curr_adam = tf.keras.optimizers.Adam(lr=lmbda)
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=[r2_keras])
    print(model.summary())

    # Parameters for the model
    batch_size = 32
    epochs = 10

    history = model.fit(
            normalize_image_data(images[train_idx]),
            positions[train_idx,:2],
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[cb_earlystopping, cb_save]
            )

    # Predict and save predictions to go with the rest of the test data.
    y_pred = model.predict(normalize_image_data(images[test_idx]))
    print("Out of sample r2:", r2_keras(positions[test_idx, :2], y_pred))
    #np.save("test_y_pred_1M.npy", y_pred)

