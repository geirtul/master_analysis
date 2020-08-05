from master_scripts.data_functions import (import_real_data,
                                           normalize_image_data)
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data import

config = {
    'DATA_PATH': "../../data/real/",
    'DATA_FILENAME': "anodedata_500k.txt",
    'MODEL_PATH': "../../models/",
    'OUTPUT_PATH': "../../data/output/",
    'CLASSIFIER': "4557cfeefc83.h5",
}


events, images = import_real_data(
    config["DATA_PATH"] + config["MODEL_PATH"])  # Images not normalized
images = normalize_image_data(images)  # Normalize images

descriptors = list(
    set([event['event_descriptor'] for event in events.values()])
)

# Load models
classifier = tf.keras.models.load_model(
    config['MODEL_PATH'] + config['CLASSIFIER']
)

# Classify events
event_classification = classifier.predict(images).argmax(axis=-1)
for event_id in events.keys():
    if event_classification[events[event_id]['image_idx']] == 0:
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
    bins=np.arange(bin_min, bin_max + 2),
    align='left'
)
ax.set_xticks(np.arange(bin_max + 2))
ax.set_xlabel("Event descriptor")
ax.set_ylabel("Number of events")
ax.legend()
fig.savefig("num_events_per_descriptor.pdf")

translate_descriptor = {
    1: "Implant",
    2: "Decay",
    3: "implant + Decay",
    4: "Light ion",
    5: "Implant + Light Ion",
    6: "Decay + Light Ion",
    7: "Implant + Decay + Light Ion",
    8: "Double (time)",
    9: "Implant + Double (time)",
    10: "Decay + Double (time)",
    11: "Implant + Decay + Double (time)",
    12: "Light ion + Double (time)",
    13: "Implant + Light Ion + Double (time)",
    14: "Decay + Light ion + Double (time)",
    15: "Implant + Decay + Light Ion + Double (time)",
    16: "Double (space)",
    17: "Implant + Double (space)",
    18: "Decay + Double (space)"
}
# Print a table-like structure for viewing
print("Classification results:")
print("|Event descriptor | Event type                   | singles | doubles |")
print("| :---           |  :---:                       | :---:   | :---:   |")
for d in descriptors:
    print("|{:^17d}|{:^30s}|{:^9d}|{:^9d}|".format(
        d,
        translate_descriptor[d],
        desc_class['single'].count(d),
        desc_class['double'].count(d)))
# Store the events as a json file
out_filename = config['OUTPUT_PATH']
+ "events_classified_"
+ config["DATA_FILENAME"][:-4]
+ "_"
+ config["CLASSIFIER"]
+ ".json"

with open(out_filename, 'w') as fp:
    json.dump(events, fp, sort_keys=True, indent=4)
print("Output file to: ", out_filename)
