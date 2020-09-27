# Imports
from master_scripts.classes import Experiment, DSNT
from master_scripts.data_functions import (normalize_image_data,
                                           normalize_position_data,
                                           event_indices,
                                           get_tf_device,
                                           get_git_root)
from master_scripts.analysis_functions import dsnt_mse
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
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
    'data': "200k",
}

# ================== Import Data ==================
DATA_PATH = get_git_root() + "data/simulated/"
images = np.load(DATA_PATH + f"images_{config['data']}.npy")
images = images.reshape(images.shape[0], 16, 16, 1)
positions = np.load(DATA_PATH + "positions_200k.npy")
labels = np.load(DATA_PATH + "labels_200k.npy")

single_indices, double_indices, close_indices = event_indices(positions)
# log-scale the images if desireable
config['scaling'] = "minmax"
if "np.log" in config['scaling']:
    images = np.log1p(images)
# set tf random seed
tf.random.set_seed(config['random_seed'])
search_name = "regression_pos_single_norm_seeded_dsnt"
with tf.device(get_tf_device(20)):
    padding = 'same'
    inputs = tf.keras.Input(shape=(16, 16, 1))
    inputs = Conv2D(32, kernel_size=(3, 3), activation='relu',
                    padding=padding)(inputs)
    #inputs = Conv2D(64, (3, 3), activation='relu', padding=padding)(inputs)
    outputs = Conv2D(64, (3, 3), activation='relu', padding=padding)(inputs)
    #outputs = DSNT()(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        dsnt_mse,
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
