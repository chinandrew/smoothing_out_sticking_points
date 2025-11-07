import numpy as np


def generate_block_autocorrelated_design(n, p, num_blocks, alpha, var):
    assert p % num_blocks == 0
    block_size = int(p / num_blocks)
    X = np.full((n, p), np.nan)
    for i in range(p):
        if i % block_size == 0:
            X[:, i] = np.random.normal(size=n, scale=np.sqrt(var))
        else:
            X[:, i] = alpha * X[:, (i - 1)] + np.sqrt(1 - alpha ** 2) * np.random.normal(size=n, scale=np.sqrt(var))
    return X
