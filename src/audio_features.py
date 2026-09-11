"""Audio feature extraction utilities.

Contains preprocessing helpers used for FMA acoustic feature preparation.
"""

import numpy as np

def normalize_features(features):
    return (features - np.mean(features, axis=0)) / (np.std(features, axis=0)+1e-8)
