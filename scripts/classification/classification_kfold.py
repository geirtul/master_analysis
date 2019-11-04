# Imports
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from analysis_functions.function import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split, StratifiedKFold


DATA_PATH = "../../data/simulated/"
OUTPUT_PATH = "../../data/output/"
MODEL_PATH = OUTPUT_PATH + "models/"

images = np.load(DATA_PATH + "images_noscale_200k.npy")
energies = np.load(DATA_PATH + "energies_noscale_200k.npy")
positions = np.load(DATA_PATH + "positions_noscale_200k.npy")
labels = np.load(DATA_PATH + "labels_noscale_200k.npy")

images = normalize_image_data(images)
n_classes = len(np.unique(labels))

net = "Project"

with tf.device('/job:localhost/replica:0/task:0/device:GPU:3'):
    # Params for k-fold cross-validation
    k_splits = 5
    k_shuffle = True
    epochs = 5
    batch_size = 32
    kfold_labels = labels.argmax(axis=-1) # StratifiedKFold doesn't take one-hot

    # Store accuracy for each fold for all models
    k_fold_results = {}
    k_fold_results[net] = []

    # Store indices for double events that are never classified correctly by
    # any model
    never_correct = []
    tmp_never_correct = []

    # Create KFold data generator
    skf = StratifiedKFold(n_splits=k_splits, shuffle=k_shuffle)
    
    # Run k-fold cv
    for train_index, test_index in skf.split(images, kfold_labels):
        single_indices, double_indices, close_indices = event_indices(positions[test_index])
        
        # Build model
        model = project_model()

        # Setup callback for saving models
        fpath = MODEL_PATH + net + "-{val_acc:.2f}.hdf5"
        cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=fpath, 
                monitor='val_acc', 
                save_best_only=True
                )

        # Train model
        history = model.fit(
            images[train_index], 
            labels[train_index], 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=(images[test_index], labels[test_index]),
            callbacks=[cb]
            )
        
        # Store the accuracy
        k_fold_results[net].append(history.history['val_acc'])

        # Get indices for wrongly classified double events
        tmp_pred = model.predict(images[test_index])
        tmp_results = tmp_pred.argmax(axis=-1).reshape(tmp_pred.shape[0], 1)
        wrong_close = test_index[close_indices][np.where(tmp_results[close_indices] == 0)[0]]


        # Add wrongly classified double event indices to tmp storage
        tmp_never_correct = tmp_never_correct + wrong_close.tolist()

        # Delete models and temporary arrays to free memory
        del(model)
        del(cb)
        del(history)
        del(tmp_pred)
        del(tmp_results)


    # Set never_correct to the common elements of wrong_close events
    never_correct = set(tmp_never_correct)

# Write results to file
with open(OUTPUT_PATH+"kfold_results.txt", "w") as resultfile:
    for key in k_fold_results.keys():
        accs = calc_kfold_accuracies(k_fold_results[key])
        resultfile.write("{}: acc_min = {:.2f} | acc_max = {:.2f} | acc_mean = {:.2f}\n".format(
            key, accs[0], accs[1], accs[2])
            )

# Write never_correct to file for later exploration
with open(OUTPUT_PATH+"never_correct_indices.txt", "w") as indexfile:
    for index in never_correct:
        indexfile.write(str(index) + "\n")

