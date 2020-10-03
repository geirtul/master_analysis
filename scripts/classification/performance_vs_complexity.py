# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data, get_tf_device,
                                           get_git_root,
					   normalize_image_data_elementwise)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, MaxPooling2D,
                                     Dropout)
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
    'data': {
        'images': "images_full.npy",
        'labels': "labels_full.npy",
    },
}

# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + config['data']['images'])
images = images.reshape(images.shape[0], 16, 16, 1)
labels = np.load(DATA_PATH + config['data']['labels'])

# log-scale the images if desireable
config['scaling'] = "minmax_elementwise"
if "np.log" in config['scaling']:
    images = np.log1p(images)

# set tf random seed
tf.random.set_seed(config['random_seed'])
experiments = {}
with tf.device(get_tf_device(20)):
    models = {}
    # Logistic
    model = Sequential()
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['logistic'] = model

    # Small Dense network
    model = Sequential()
    model.add(Dense(10, activation='relu'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['dense_10'] = model

    # Large dense network
    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['dense_100_5'] = model

    # Small convolutional network
    model.add(Conv2D(10, activation='relu'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    models['cnn_shallow'] = model
    
    # Large convolutional network, deeper and wider
    model.add(Conv2D(32, activation='relu'))
    model.add(Conv2D(32, activation='relu'))
    model.add(Conv2D(64, activation='relu'))
    model.add(Conv2D(64, activation='relu'))
    model.add(Conv2D(128, activation='relu'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    models['cnn_deepwide'] = model

    # Run experiments
    for k, m in models.items():
        experiment = Experiment(
            model=m,
            config=config,
            model_type="classification",
            experiment_name="full_training_classifier_" + k
        )
        if "logistic" in k or "dense" in k:
            experiment.run(
            normalize_image_data(images).reshape(images.shape[0], 256),
            labels,
        )
        else:
            experiment.run(
                normalize_image_data(images),
                labels,
            )
        experiment.save()
        mpath = experiment.config['path_args']['models'] + experiment.id + ".h5"
        model.save(mpath)
        experiments[k] = experiment.id
for k, v in experiments.items():
    print(k, ":", v)
