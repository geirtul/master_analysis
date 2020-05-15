import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys
import json
import matplotlib.pyplot as plt
from master_data_functions.functions import *
from analysis_functions.functions import *
from tqdm import tqdm

# ## Data import

config = {
    "DATA_PATH": "../../data/real/",
    "DATA_FILENAME": "anodedata_sample.txt",
    "SAMPLE_PATH": "../../data/sample/CeBr10k_1.txt",
    "MODEL_PATH": "../../data/output/models/",   
    "OUTPUT_PATH": "../../data/output/",
    "CLASSIFIER": "Project-0.97.hdf5",                      
    "SINGLE_ENERGY_MODEL": "cnn_energy_single_r2_0.97.hdf5",    
    "SINGLE_POSITION_MODEL": "cnn_pos_single_r2_0.99.hdf5",
    "DOUBLE_ENERGY_MODEL": "double_energy_model_name.hdf5",    
    "DOUBLE_POSITION_MODEL": "double_position_model_name.hdf5" 
}


events, images = import_real_data(config) # images not normalized
images = normalize_image_data(images)

descriptors = list(set([event['event_descriptor'] for event in events.values()]))
# ## Load models
classifier = tf.keras.models.load_model(config["MODEL_PATH"] + config["CLASSIFIER"])

# ## Classify events
event_classification = classifier.predict(images).argmax(axis=-1)
for event_id in events.keys():
    if event_classification[events[event_id]["image_idx"]] == 0:
        events[event_id]['event_class'] = "single" 
    else:
        events[event_id]['event_class'] = "double"


desc_class = {
    'single': [],
    'double': [],
}
for event in events.values():
    desc_class[event['event_class']].append(event['event_descriptor'])

bin_min = np.amin(np.array(descriptors))
bin_max = np.amax(np.array(descriptors))

fig, ax = plt.subplots()
ax.hist(
    [desc_class['single'], desc_class['double']], 
    histtype='barstacked', 
    label=["single", "double"],
    bins=np.arange(bin_min, bin_max+2),
    align='left'
)
ax.set_xticks(np.arange(bin_max+2))
ax.set_xlabel("Event descriptor")
ax.set_ylabel("Number of events")
ax.legend()
fig.savefig("num_events_per_descriptor.pdf")

# Print a table-like structure for viewing
print("Classification results:")
print("Event descriptor | singles | doubles |")
for d in descriptors:
    print("{:^17d}|{:^9d}|{:^9d}|".format(
        d, 
        desc_class['single'].count(d), 
        desc_class['double'].count(d)
        )
        )
# Store the events as a json file
out_filename = config['OUTPUT_PATH'] + "events_classified_" + config["DATA_FILENAME"][:-4] + ".json"
with open(out_filename, 'w') as fp:
    json.dump(events, fp, sort_keys=True, indent=4)

