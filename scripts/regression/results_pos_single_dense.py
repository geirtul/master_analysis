# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data,
                                           normalize_position_data,
                                           event_indices,
                                           get_tf_device,
                                           get_git_root)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import tensorflow as tf
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
with open("results_experiment_config.json", 'r') as fp:
    config = json.load(fp)

# ================== Callbacks ====================


# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + config['data']['images'])
images = images.reshape(images.shape[0], 16, 16, 1)
labels = np.load(DATA_PATH + config['data']['labels'])

# log-scale the images if desireable
config['scaling'] = "minmax"
if "np.log" in config['scaling']:
    images = np.log1p(images)

# set tf random seed
tf.random.set_seed(config['random_seed'])
# ================== Import Data ==================
with tf.device(get_tf_device(20)):
    padding = 'same'
    model = Sequential()
    model.add(Conv2D(10, kernel_size=(3, 3),
                     activation='relu', input_shape=(16, 16, 1),
                     padding=padding))
    model.add(Conv2D(64, (3, 3), activation='relu', padding=padding))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding=padding))
    model.add(Conv2D(64, (3, 3), activation='relu', padding=padding))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(
        loss='mse',
        optimizer='adam',
    )
    print(model.summary())

    # Run experiment
    experiment = Experiment(
        model=model,
        config=config,
        model_type="regression",
        experiment_name=search_name
    )
    experiment.run(
        normalize_image_data(images[single_indices]),
        normalize_position_data(positions[single_indices])[:, :2],
    )
    experiment.save()
    mpath = experiment.config['path_args']['models'] + experiment.id + ".h5"
    model.save(mpath)
