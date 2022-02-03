# Imports
from master_scripts.classes import Experiment
from master_scripts.data_functions import (normalize_image_data,
                                           normalize_position_data,
                                           event_indices,
                                           get_tf_device,
                                           get_git_root)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from master_scripts.models_pretrained import pretrained_model
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
energies = np.load(DATA_PATH + config['data']['energies'])
positions = np.load(DATA_PATH + config['data']['positions'])

single_indices, double_indices, close_indices = event_indices(positions)
# log-scale the images if desireable
config['scaling'] = "minmax"
if "np.log" in config['scaling']:
    images = np.log1p(images)

# set tf random seed
tf.random.set_seed(config['random_seed'])

# For the pretrained vgg model we have to use data with 3 channels.
# This is solved by concatenating the images with themselves
images = np.concatenate((images, images, images), axis=-1)

with tf.device(get_tf_device(20)):
    model = pretrained_model("VGG16", input_dim=(16, 16, 3), trainable=False)
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
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
        experiment_name="results_energies_single_pretrained",
    )
    experiment.run_kfold(
        images[single_indices],
        energies[single_indices, 0],
    )
    experiment.save(save_model=False, save_indices=False)
    # print("Finished experiment:", experiment.id, " named ",
    #      experiment.experiment_name, "with data ",
    #      config['data']['images'])

    # Fine tune the model with new experiment.
    model_tune = pretrained_model(
        "VGG16", input_dim=(16, 16, 3), trainable=True)
    model_tune.add(model.layers[-2])
    model_tune.add(model.layers[-1])
    model_tune.compile(
        optimizer=tf.keras.optimizers.Adam(0.00001),
        loss='mse',
    )
    config_tune = config
    config_tune['fit_args']['epochs'] = 1
    config_tune['compile_args']['adam_lr'] = 1E-5
    experiment_tune = Experiment(
        model=model_tune,
        config=config_tune,
        model_type="regression",
        experiment_name="results_energies_single_pretrained_finetune",
    )
    experiment_tune.run_kfold(
        images[single_indices],
        energies[single_indices, 0],
    )
    experiment_tune.save(save_model=True, save_indices=False)
    print("Finished experiment:", experiment_tune.id, " named ",
          experiment_tune.experiment_name, "with data ",
          config_tune['data']['images'])
