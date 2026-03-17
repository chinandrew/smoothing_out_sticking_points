import numpy as np
import pickle
from scipy.special import expit


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


n = 2000
p = 2000
nonzero_coefs = 20
num_blocks = 20
sigma2 = 100

for alpha in [0.5, 0.9, 0.99]:
    # linreg data
    X_scale2 = 1
    np.random.seed(0)
    z_true = np.zeros(p)
    z_true[np.random.choice(range(p), size=nonzero_coefs, replace=False)] = 1
    beta_true = 2 * np.random.binomial(1, 0.5, size=p) - 1
    assert p % num_blocks == 0
    assert sum(z_true) == nonzero_coefs

    X = generate_block_autocorrelated_design(n, p, num_blocks, alpha, X_scale2)

    y = X @ (z_true * beta_true) + np.random.normal(scale=np.sqrt(sigma2), size=n)
    with open(f"linreg_simulation/simulated_data_a{str(alpha).replace('.', '-')}.p", "wb") as f:
        pickle.dump((X, y, z_true, beta_true, sigma2), f)

    # logreg data
    np.random.seed(0)
    X_int = np.hstack((np.ones((len(X), 1)), X))
    beta_int = np.insert(beta_true * z_true, 0, -8)  # leads to ~5% prevalance

    y = np.random.binomial(1, expit(X_int @ beta_int))
    with open(f"logreg_simulation/simulated_data_a{str(alpha).replace('.', '-')}_logreg.p", "wb") as f:
        pickle.dump((X_int, y, beta_int), f)
