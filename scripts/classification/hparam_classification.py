# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import normalize_image_data, get_tf_device
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 1,
        'batch_size': 32,
    },
    'random_seed': 120,
}

# ================== Import Data ==================
images = np.load(DATA_PATH + "images_200k.npy")
indices = np.random.choice(
    np.arange(images.shape[0]),
    10000,
    replace=False)
images = images[indices]
images = images.reshape(images.shape[0], 16, 16, 1)
labels = np.load(DATA_PATH + "labels_200k.npy")[indices]

x_idx = np.arange(images.shape[0])
train_idx, val_idx, u1, u2 = train_test_split(
    x_idx, x_idx, random_state=config['random_seed']
)

# ================== Search params ================
num_filters = [4, 8]
kernel_sizes = [(3, 3), (5, 5)]  # , (7, 7), (9, 9)]
dense_sizes = [32, 64, 128, 256]

with tf.device(get_tf_device(20)):
    # Build models
    models = []
    for n_filters in num_filters:
        for k_size in kernel_sizes:
            model = Sequential()
            model.add(
                Conv2D(
                    filters=n_filters,
                    kernel_size=k_size,
                    input_shape=images.shape[1:],
                    padding='same',
                    activation='relu',
                )
            )
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))

            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            models.append(model)

    # Run experiments
    for model in models:
        experiment = Experiment(
            experiment_type="hparam_classification",
            model=model,
            config=config,
        )
        experiment.run(
            normalize_image_data(images[train_idx]),
            labels[train_idx],
            normalize_image_data(images[val_idx]),
            labels[val_idx],
        )
        experiment.save()
