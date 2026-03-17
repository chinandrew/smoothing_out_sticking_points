"""
Run sticky samplers.

Takes in the following position command line arguments in order.
Expects design matrice generated from ../linreg_simulation/run_simulation.py from
linear regression example.

method: "zz", "laplace", or "gaussian"
iters: Integer number of samples to draw
thin: Amount to thin samples
seed: Seed
step_size: Step size for numerical integration. Perturbed by 20% in either direction.
num_steps: Minimum number of steps to take for numerical integration.
    Actual value is then drawn as Uniform(num_steps, 3*num_steps)
true_init: Whether to init at random or from true coefficients + perturbation. 1 or 0.
p_slab: Prior slab probability
alpha: Block correlation in design matrix, only used to load data and save output
tau2: Prior coefficient variance (normal prior)

Example command to run the numerically integrated unadjusted zig-zag:
`python run_logreg_simulation.py zz 100000 1 1 0.02 50 1 0.01 0.9`
"""

import cProfile
import pickle
import pstats
import socket
import sys
import time
import os

import numpy as np
from scipy.special import expit

sys.path.append('.')

from likelihood import logdensity, grad_logdensity
from momentum import Gaussian, Laplace
from sampler import hmc, zz, compute_latent_width

method = sys.argv[1]
iters = int(sys.argv[2])
thin = int(sys.argv[3])
seed = int(sys.argv[4])
step_size = float(sys.argv[5])
num_steps = int(sys.argv[6])
p_slab = float(sys.argv[7])
alpha = float(sys.argv[8])
node = socket.gethostname().split(".")[0]

with open(f"logreg_simulation/simulated_data_a{str(alpha).replace('.', '-')}_logreg.p", "rb") as f:
    X_int, y, beta_true = pickle.load(f)
true_idxs = np.where(beta_true)[0]

num_covs = X_int.shape[1] - 1
tau = 1
tau2_vector = np.array([np.inf] + [tau ** 2] * num_covs)
p_slab_vector = np.array([1] + [p_slab] * num_covs)
boundary = compute_latent_width(p_slab_vector, tau2_vector) / 2

print(seed, num_steps, method)
np.random.seed(seed)
beta_init = np.random.uniform(-boundary, boundary, len(beta_true))
beta_init[true_idxs] = beta_true[true_idxs] + np.sign(beta_true[true_idxs]) * boundary[true_idxs] + np.random.randn(
    len(true_idxs)) * 0.001
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

filename = (f"{method}_"
            f"{str(alpha).replace('.', '-')}_"
            f"{str(p_slab).replace('.', '-')}_"
            f"seed{seed}_"
            f"steps{num_steps}_"
            f"stepsize{str(step_size).replace('.', '-')}_"
            f"iters{iters}_"
            f"{node}")

pstats.Stats(profiler).dump_stats(
    f"{filename}.prof",
)
runtime = time.perf_counter() - start
meta["runtime"] = runtime
print(meta)
with open(f"{filename}.p", "wb") as f:
    pickle.dump((samples, meta), f)
