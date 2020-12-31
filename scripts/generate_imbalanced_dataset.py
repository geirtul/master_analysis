import warnings
from master_scripts.data_functions import (generate_imbalanced_dataset_indices,
                                           get_git_root)
import numpy as np
warnings.filterwarnings('ignore', category=FutureWarning)

repo_root = get_git_root()
images = np.load(repo_root + "data/simulated/images_full_pixelmod.npy")
positions = np.load(repo_root + "data/simulated/positions_full.npy")
energies = np.load(repo_root + "data/simulated/energies_full.npy")
labels = np.load(repo_root + "data/simulated/labels_full.npy")

imba_indices = generate_imbalanced_dataset_indices(positions, 0.05)

np.save(repo_root + "data/simulated/images_full_pixelmod_imbalanced.npy",
        images[imba_indices])
np.save(repo_root + "data/simulated/positions_full_imbalanced.npy",
        positions[imba_indices])
np.save(repo_root + "data/simulated/energies_full_imbalanced.npy",
        energies[imba_indices])
np.save(repo_root + "data/simulated/labels_full_imbalanced.npy",
        labels[imba_indices])
