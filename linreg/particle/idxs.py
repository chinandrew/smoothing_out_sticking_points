"""Functions related to finding specific indices of vectors."""
import numpy as np


def min_argmin(x: np.ndarray):
    """Find minimum value and index of minimum of a vector. Returns first index in case of tie."""
    if len(x) == 0:
        return np.inf, np.nan
    idx = np.argmin(x)
    return x[idx], idx


def convert_to_full_idx(bool_vector: np.ndarray,
                        i: int):
    """
    Get index of ith True in a boolean vector.

    If bool_vector = [True, False, False, True], the 1-indexed True is at index 3, i.e.
    convert_to_full_idx(bool_vector, 1) == 3
    """
    if np.isnan(i):
        return i
    return np.where(bool_vector)[0][i]


def convert_to_sub_idx(bool_vector: np.ndarray,
                       i: int):
    """
    Turn index of a True value in a boolean vector into the index of the vector of only True's.

    If bool_vector = [True, False, False, True], the 3-indexed value is the 1-indexed True.
    convert_to_full_idx(bool_vector, 3) == 1.

    If `bool_vector[i]` is not True, will error.
    """
    if np.isnan(i):
        return i
    return np.where(np.where(bool_vector)[0] == i)[0][0]
