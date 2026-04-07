import pandas as pd
from scipy.stats import ks_2samp

def detect_drift(train_path, new_path):
    train = pd.read_csv(train_path)
    new = pd.read_csv(new_path)

    drift_scores = {}

    for col in ["feature1", "feature2"]:
        stat, p_value = ks_2samp(train[col], new[col])
        drift_scores[col] = p_value

    return drift_scores