"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def standardize_zero_to_one(x: np.ndarray) -> np.ndarray:
    """Standardize each column from 0 to 1."""
    max_ = np.max(x, axis=0)
    min_ = np.min(x, axis=0)
    x = (x - min_) / (max_ - min_)
    return x


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def random_split_data(
    x: np.ndarray, y: np.ndarray, ratio=0.8, seed=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split data into training and testing sets according to the given ratio."""
    num_samples = x.shape[0]
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    split_point = int(num_samples * ratio)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    return x[train_indices], y[train_indices], x[test_indices], y[test_indices]


def random_k_fold(
    x: np.ndarray, y: np.ndarray, k: int, seed=None
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Randomly split data into k folds for cross-validation."""
    num_samples = x.shape[0]
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(num_samples)
    fold_sizes = (num_samples // k) * np.ones(k, dtype=int)
    fold_sizes[: num_samples % k] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        folds.append(
            (x[train_indices], y[train_indices], x[test_indices], y[test_indices])
        )
        current = stop
    return folds


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def negative_reduction(y, tx, reduction_factor):
    """
    Reduces the influence of negative samples in the dataset.
    :param y: Labels {0, 1} form
    :param tx: Features
    :param reduction_factor: Factor by which to reduce negative samples
    :return: Reduced y and tx
    """
    if reduction_factor >= 1.0:
        return y, tx
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]
    num_negatives_to_keep = int(len(negative_indices) * reduction_factor)
    selected_negative_indices = np.random.choice(
        negative_indices, num_negatives_to_keep, replace=False
    )
    selected_indices = np.concatenate((positive_indices, selected_negative_indices))
    # print(f"Percent of positive samples: {100*len(positive_indices)/(len(positive_indices) + num_negatives_to_keep):.2f}%")
    return y[selected_indices], tx[selected_indices]


def sort_abs_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sorts the features by absolute correlation with the labels
    :param x: numpy array of shape=(N, D) features
    :param y: numpy array of shape=(N, ) labels
    :return: numpy array of shape=(D, ) sorted feature indices
    """
    corrs = np.array([np.corrcoef(x[:, i], y)[0, 1] for i in range(x.shape[1])])
    abs_corrs = np.abs(corrs)
    sorted_indices = np.argsort(-abs_corrs)
    return sorted_indices
