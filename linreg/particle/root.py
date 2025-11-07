"""Functions related to roots of equations."""
import numpy as np


def min_root(a, b, c, lower_bound=0):
    """Find smallest root which is greater than `lower_bound` for a ax^2+bx+c=0."""
    disc = b ** 2 - 4 * a * c
    has_root = disc > 0
    root = np.full((2, len(c)), np.inf)
    aa = 2 * a[has_root]
    sqrtdisc = np.sqrt(disc[has_root])
    root[0, has_root] = (- b[has_root] - sqrtdisc) / aa
    root[1, has_root] = (sqrtdisc - b[has_root]) / aa
    root[root <= lower_bound] = np.inf
    return np.min(root, axis=0)
