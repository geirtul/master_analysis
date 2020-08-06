# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data, get_tf_device,
                                           get_git_root)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 20,
        'batch_size': 64,
    },
    'random_seed': 120,
}


# ================== Callbacks ====================


# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + "images_full.npy")
images = images.reshape(images.shape[0], 16, 16, 1)
labels = np.load(DATA_PATH + "labels_full.npy")

x_idx = np.arange(images.shape[0])
train_idx, val_idx, u1, u2 = train_test_split(
    x_idx, x_idx, random_state=config['random_seed']
)

# log-scale the images if desireable
# images = np.log1p(images)
config['scaling'] = "minmax"

# set tf random seed
tf.random.set_seed(config['random_seed'])
with tf.device(get_tf_device(20)):
    # Build model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=(16, 16, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Run experiment
    experiment = Experiment(
        model=model,
        config=config,
        model_type="classification",
        experiment_name="full_training_classifier_no_regu_deeper"
    )
    experiment.run(
        normalize_image_data(images[train_idx]),
        labels[train_idx],
        normalize_image_data(images[val_idx]),
        labels[val_idx],
    )
    experiment.save()
    mpath = experiment.config['path_args']['models'] + experiment.id + ".h5"
    model.save(mpath)
