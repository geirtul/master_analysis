# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data, get_tf_device,
                                           get_git_root)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 20,
        'batch_size': 32,
    },
    'random_seed': 120,
}

# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + "images_200k.npy")
images = images.reshape(images.shape[0], 16, 16, 1)
labels = np.load(DATA_PATH + "labels_200k.npy")

x_idx = np.arange(images.shape[0])
train_idx, val_idx, u1, u2 = train_test_split(
    x_idx, x_idx, random_state=config['random_seed']
)

# ================== Search params ================
num_filters = [4, 8, 16, 32, 64]
dense_sizes = [32, 64, 128, 256]

# set tf random seed
tf.random.set_seed(config['random_seed'])
id_param = {}
search_name = "architecture_search_seeded_2conv"
with tf.device(get_tf_device(20)):
    # Build models
    models = []
    for n_filters1 in num_filters:
        for n_filters2 in num_filters:
            for d_size in dense_sizes:
                model = Sequential()
                model.add(
                    Conv2D(
                        filters=n_filters1,
                        kernel_size=(3, 3),
                        input_shape=images.shape[1:],
                        padding='same',
                        activation='relu',
                    )
                )
                model.add(MaxPool2D(padding='same'))
                model.add(
                    Conv2D(
                        filters=n_filters2,
                        kernel_size=(3, 3),
                        input_shape=images.shape[1:],
                        padding='same',
                        activation='relu',
                    )
                )
                model.add(Flatten())
                model.add(Dense(d_size, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))

                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                models.append(model)

                # Run experiment
                experiment = Experiment(
                    model=model,
                    config=config,
                    model_type="classification",
                    experiment_name=search_name
                )
                experiment.run(
                    normalize_image_data(images[train_idx]),
                    labels[train_idx],
                    normalize_image_data(images[val_idx]),
                    labels[val_idx],
                )
                experiment.save()
                id_param[experiment.experiment_id] = {
                    'l1_filters': n_filters1,
                    'l2_filters': n_filters2,
                    'kernel_size': (3, 3),
                    'dense_size': d_size,
                }
search_path = get_git_root() + "experiments/searches/"
with open(search_path + search_name + ".json", "w") as fp:
    json.dump(id_param, fp, indent=2)
