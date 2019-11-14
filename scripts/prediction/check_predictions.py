import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import sys
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from master_models.prediction import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend


# ================== Custom Functions ==================
# Define R2 score for metrics since it's not available by default
def r2_keras(y_true, y_pred):
    SS_res =  backend.sum(backend.square(y_true - y_pred)) 
    SS_tot = backend.sum(backend.square(y_true - backend.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + backend.epsilon()) )

# Load model
MODEL_PATH = "../../data/output/models/"
FIGURE_PATH = "../../"
DATA_PATH = "../../data/simulated/"

# Load data
images = np.load(DATA_PATH + "images_1M.npy")
positions = np.load(DATA_PATH + "positions_1M.npy")
energies = np.load(DATA_PATH + "energies_1M.npy")
with tf.device('/GPU:2'):
    name = "cnn-r2-0.79.hdf5"
    loaded_model = tf.keras.models.load_model(MODEL_PATH+name, custom_objects={'r2_keras': r2_keras})
    
    
    y_pred = loaded_model.predict(images)
    y_resid = y_pred - positions
    
single, double, close = event_indices(positions)
rel_dist = relative_distance(positions)
rel_E = relative_energy(energies)

y_resid = np.sum(y_resid, axis=1)
indices = np.random.choice(double, 10000, replace=False)

plt.scatter(rel_E[indices], y_resid[indices], alpha=0.2)
plt.xlabel("rel_E")
plt.ylabel("sum_residuals")
plt.savefig("residuals_rel_e.pdf")
plt.clf()
    
plt.scatter(rel_dist[indices], y_resid[indices], alpha=0.2)
plt.xlabel("rel_dist")
plt.ylabel("sum_residuals")
plt.savefig("residuals_rel_dist.pdf")
# Plot
#fig, ax = plt.subplots(2,2)
#ax[0,0].hist(y_pred[double,0], bins=25)
#ax[0,1].hist(y_pred[double,1], bins=25)
#ax[1,0].hist(y_pred[double,2], bins=25)
#ax[1,1].hist(y_pred[double,3], bins=25)
#fig.savefig("predictions.pdf")
#fig.clf()
#
#fig, ax = plt.subplots(2,2)
#ax[0,0].hist(y_resid[double,0], bins=25)
#ax[0,1].hist(y_resid[double,1], bins=25)
#ax[1,0].hist(y_resid[double,2], bins=25)
#ax[1,1].hist(y_resid[double,3], bins=25)
#fig.savefig("residuals.pdf")
    
