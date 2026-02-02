import cProfile
import pickle
import pstats
import socket
import sys
import time
import os

import numpy as np
from scipy.special import expit

from likelihood import logdensity, grad_logdensity
from momentum import Gaussian, Laplace
from sampler import hmc, zz, compute_latent_width

method = sys.argv[1]
iters = int(sys.argv[2])
thin = int(sys.argv[3])
seed = int(sys.argv[4])
step_size = float(sys.argv[5])
num_steps = int(sys.argv[6])
true_init = bool(int(sys.argv[7]))
p_slab = float(sys.argv[8])
alpha = float(sys.argv[9])
node = socket.gethostname().split(".")[0]

with open(f"simulated_data_a{alpha}.p", "rb") as f:  # Pickled tuple of data from /simulation/run_simulation.py
    X, _, z_true, beta = pickle.load(f)
beta = beta * z_true
np.random.seed(0)
X_int = np.hstack((np.ones((len(X), 1)), X))
beta_int = np.insert(beta, 0, -8)  # leads to ~5% prevalance
true_idxs = np.where(beta_int)[0]

y = np.random.binomial(1, expit(X_int @ beta_int))

if not os.path.exists(f"simulated_data_a{alpha}_logreg.p"):
    with open(f"simulated_data_a{alpha}_logreg.p", "wb") as f:
        pickle.dump((X_int, y, true_idxs, beta_int), f)
else:
    pass

tau = 1
tau2_vector = np.array([np.inf] + [tau ** 2] * X.shape[1])
p_slab_vector = np.array([1] + [p_slab] * X.shape[1])
boundary = compute_latent_width(p_slab_vector, tau2_vector) / 2

print(seed, num_steps, method)
np.random.seed(seed)
beta_init = np.random.uniform(-boundary, boundary, len(beta_int))
if true_init:
    beta[true_idxs] + np.sign(beta[true_idxs]) * boundary[true_idxs] + np.random.randn(len(true_idxs))*0.001
    beta_init[0] = beta_int[0] + np.random.randn() * 0.001
start = time.perf_counter()
profiler = cProfile.Profile()
profiler.enable()
if method == "zz":
    samples, meta = zz(iters * num_steps, beta_init, X_int, y, tau2_vector, p_slab_vector, logdensity, grad_logdensity,
                       step_size, num_steps * thin, adjust=False)
elif method == "laplace":
    samples, meta = hmc(iters, beta_init, X_int, y, tau2_vector, p_slab_vector, logdensity, grad_logdensity, Laplace,
                        num_steps, step_size, thin)
elif method == "gaussian":
    samples, meta = hmc(iters, beta_init, X_int, y, tau2_vector, p_slab_vector, logdensity, grad_logdensity, Gaussian,
                        num_steps, step_size, thin)
else:
    raise NotImplementedError
profiler.disable()
pstats.Stats(profiler).dump_stats(
    f"{method}_{alpha}_{pslab}_seed{seed}_steps{num_steps}_stepsize{step_size}_iters{iters}_{node}.prof",
)
runtime = time.perf_counter() - start
meta["runtime"] = runtime
print(meta)
with open(
        f"{method}_{alpha}_{pslab}_seed{seed}_steps{num_steps}_stepsize{step_size}_iters{iters}_{node}.p",
        "wb") as f:
    pickle.dump((samples, meta), f)
