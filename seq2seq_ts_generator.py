from tsaug import TimeWarp, Drift, Pool, AddNoise # , Crop, Quantize, Drift, Reverse
import numpy as np
from wfdb.processing import (
    normalize_bound
)

my_augmenter = (TimeWarp(n_speed_change=1, prob=0.33)
                + Drift(max_drift=(0.05, 0.3))
                + Pool(prob=0.2)
                + AddNoise(prob=0.2))


def tsaug_generator(X_all, y_all, batch_size):
    """
    Generate Time Series data with sequence labels
    Data generator that yields training data as batches.

    1. Randomly selects one sample from time series signals
    2. Applies time series augmentations to X and Y
    3. Normalizes result for X data
    4. Reshapes data into correct format for training


    Parameters
    ----------
    X_all : 3D numpy array
        (N, seqlen, features=1)

    y_all : 3D numpy array (binary labels for my case)
        (N, seqlen, classes=1)
    batch_size : int
        Number of training examples in the batch
    Yields
    ------
    (X, y) : tuple
        Contains training samples with corresponding labels
    """
    while True:

        X = []
        y = []

        while len(X) < batch_size:
            random_sig_idx = np.random.randint(0, X_all.shape[0])
            x1 = X_all[random_sig_idx].flatten()
            y1 = y_all[random_sig_idx].flatten()

            # Add noise into data window and normalize it again

            X_aug, y_aug = my_augmenter.augment(x1, y1)
            X_aug = normalize_bound(X_aug, lb=-1, ub=1)

            X.append(X_aug)
            y.append(y_aug)

        X = np.asarray(X)
        y = np.asarray(y)

        X = X.reshape(X.shape[0], X.shape[1], 1)
        y = y.reshape(y.shape[0], y.shape[1], 1).astype(int)

        yield (X, y)


