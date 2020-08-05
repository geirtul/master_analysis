# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data,
                                           normalize_position_data,
                                           event_indices,
                                           get_tf_device,
                                           get_git_root)
from master_scripts.models_regression import position_double_cnn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ================== Config =======================
config = {
    'fit_args': {
        'epochs': 10,
        'batch_size': 32,
    },
    'random_seed': 120,
}

# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + "images_200k.npy")
images = images.reshape(images.shape[0], 16, 16, 1)
positions = np.load(DATA_PATH + "positions_200k.npy")
labels = np.load(DATA_PATH + "labels_200k.npy")

single_indices, double_indices, close_indices = event_indices(positions)
train_idx, val_idx, u1, u2 = train_test_split(
    double_indices, double_indices, random_state=config['random_seed']
)

# set tf random seed
tf.random.set_seed(config['random_seed'])
id_param = {}
search_name = "test_position_regression_double_seeded"
with tf.device(get_tf_device(20)):
    model = position_double_cnn()
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
        normalize_image_data(images[train_idx]),
        normalize_position_data(positions[train_idx]),
        normalize_image_data(images[val_idx]),
        normalize_position_data(positions[val_idx]),
    )
    experiment.save()
    id_param[experiment.experiment_id] = {}
search_path = get_git_root() + "experiments/searches/"
with open(search_path + search_name + ".json", "w") as fp:
    json.dump(id_param, fp, indent=2)
