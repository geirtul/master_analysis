# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import normalize_image_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# PATH variables
DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

# ================== Import Data ==================
img_train = np.load(DATA_PATH + "images_200k_train.npy")
img_val = np.load(DATA_PATH + "images_200k_val.npy")
y_train = np.load(DATA_PATH + "labels_200k_train.npy")[:10]
y_val = np.load(DATA_PATH + "labels_200k_val.npy")

# Reshape and normalize images
img_train = normalize_image_data(
    img_train.reshape(img_train.shape[0], 16, 16, 1)
)[:10]
img_val = normalize_image_data(
    img_val.reshape(img_val.shape[0], 16, 16, 1)
)

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 1,
        'batch_size': 32,
    },
}

# ================== Search params ================
num_filters = [4, 8]
kernel_sizes = [(3, 3), (5, 5)]  # , (7, 7), (9, 9)]
dense_sizes = [32, 64, 128, 256]

# Build models
models = []
for n_filters in num_filters:
    for k_size in kernel_sizes:
        model = Sequential()
        model.add(Conv2D(
            n_filters, k_size, input_shape=img_train.shape[1:],
            padding='same', activation='relu')
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
    experiment.run(img_train, y_train, img_val, y_val)
    experiment.output_experiment()
