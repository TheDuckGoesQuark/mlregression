from sklearn.model_selection import train_test_split

import numpy as np


def split_data(x, y):
    bins = np.linspace(0, 586, 50)
    y_binned = np.digitize(y[:, 0], bins)

    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y_binned)
