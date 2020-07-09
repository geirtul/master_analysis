# Imports
from sklearn.model_selection import train_test_split
from master_models.classification import project_model
from master_scripts.classes import Experiment
from master_scripts.data_functions import get_tf_device, normalize_image_data
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Determine tensorflow device
MAX_LOAD = 20  # maximum GPU-util 20%
DEVICE = get_tf_device(MAX_LOAD)

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Import Data ==================
img_train = np.load(DATA_PATH + "images_200k_train.npy")
img_val = np.load(DATA_PATH + "images_200k_val.npy")
y_train = np.load(DATA_PATH + "labels_200k_train.npy")
y_val = np.load(DATA_PATH + "labels_200k_val.npy")
val_data = (normalize_image_data(img_val), y_val)

# ================== Config =======================


# ================== Search params ================
kernel_sizes = [(3, 3), (5, 5), (7, 7), (9, 9)]


with tf.device(DEVICE):
    model = project_model()

    # Callbacks to save best model
    # Setup callback for saving models
    fpath = MODEL_PATH + experiment.id
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
    model.fit(
        normalize_image_data(img_train),
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_data,
        callbacks=[cb_save, cb_earlystopping],
    )
