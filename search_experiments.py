import json
import sys
import os

# ============================================================================
# Search and sort experiment outputs by any given value present in the
# json files
# ============================================================================
if len(sys.argv) < 2:
    print("Provide a key to search for")
    exit(1)
else:
    searchkey = sys.argv[1]


def search_dict(key, obj):
    """
    https://stackoverflow.com/a/20254842
    """
    matches = []
    if hasattr(obj, 'items'):
        for k, v in obj.items():
            if k == key:
                matches.append(v)
            elif isinstance(v, dict):
                matches += search_dict(key, v)
    return matches


PATH = "experiments/"
# Get filenames
filenames = [el for el in os.listdir(PATH) if el.endswith("json")]

experiments = []
results = {}

# Load json objects
for fname in filenames:
    with open(PATH + fname, 'r') as fp:
        experiments.append(json.load(fp))

# Search each experiment for key
for e in experiments:
    found = search_dict(searchkey, e)
    results[e['experiment_id']] = found[0]

    # Sort the results by value
results_sorted = {k: results[k] for k in sorted(results, key=results.get,
                                                reverse=True)}

with open("results_" + searchkey + ".json", "w") as fp:
    json.dump(results_sorted, fp, indent=2)
