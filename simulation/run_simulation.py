"""
Run sticky samplers.

Takes in the following position command line arguments in order

method: "shzz", "szz", or "szz-constant"
n_iter: Integer number of samples to draw
seed: Seed
n: Number of observations
p: Number of covariates
nonzero_coefs: Number of true nonzero coefficients
alpha: Block correlation in design matrix
sigma2: Noise variance
thin: Amount to thin samples. NOTE: Only used by "shzz",
    for "szz" and "szz-constant" you can simply increase the integration time `t1` below.
num_blocks: Number of correlated blocks in design matrix
p_slab: Prior slab probability
tau2: Prior coefficient variance (normal prior)
t1: For "szz" and "szz-constant", discretization time for generating samples from continuous time trajectory.
    For "shzz", lower bound for travel time.
t2 ("shzz" only): For "shzz", upper end of travel time. Travel time is sampled Unif(t1, t2).


Example command to run the latent sticky sampler:
`python run_simulation.py szz-constant 25000 4 2000 2000 20 0.99 100 1 20 0.1 1 2`
"""
import pickle
import socket
import sys

import numpy as np

sys.path.append('..')

from simulation.generate_data import generate_block_autocorrelated_design
from linreg.sampler import sample

method = sys.argv[1]
n_iter = int(sys.argv[2])
seed = int(sys.argv[3])
n = int(sys.argv[4])
p = int(sys.argv[5])
nonzero_coefs = int(sys.argv[6])
alpha = float(sys.argv[7])
sigma2 = float(sys.argv[8])
thin = int(sys.argv[9])
if method != "shzz":
    thin = 1 # PDMPs just use longer discretization times
num_blocks = int(sys.argv[10])
p_slab = float(sys.argv[11])
tau2 = float(sys.argv[12])
t1 = float(sys.argv[13])
try:
    t = [t1, float(sys.argv[14])]
    t_str = "-".join([str(i).replace(".", "-") for i in t])
except IndexError:
    t = t1
    t_str = str(t).replace(".", "-")

X_scale2 = 1
np.random.seed(0)
z_true = np.zeros(p)
z_true[np.random.choice(range(p), size=nonzero_coefs, replace=False)] = 1
beta_true = 2*np.random.binomial(1, 0.5, size=p) - 1
assert p % num_blocks == 0
assert sum(z_true) == nonzero_coefs

X = generate_block_autocorrelated_design(n, p, num_blocks, alpha, X_scale2)

y = X @ (z_true * beta_true) + np.random.normal(scale=np.sqrt(sigma2), size=n)

initial_perturbed = beta_true * z_true + np.random.normal(size=p, scale=0.001)
samples_raw, aug_samples_raw, bounces, sampler, runtime = sample(
    method, n_iter, initial_perturbed, z_true, X, y, sigma2, p_slab, tau2, t, thin, seed=seed)

node = socket.gethostname().split(".")[0]

filename = (f'{method}_seed{seed}_'
            f't{t_str}_'
            f'thin{thin}_'
            f'pslab{str(p_slab).replace(".", "-")}_'
            f'tau2{str(tau2).replace(".", "-")}_'
            f'iter{n_iter}_n{n}_p{p}_'
            f'nonzero{nonzero_coefs}_'
            f'alpha{str(alpha).replace(".", "-")}_'
            f'sigma2-{str(sigma2).replace(".", "-")}_'
            f'nblock{num_blocks}_'
            f'random-init_'
            f'{node}')
with open(f"{filename}.pkl", "wb") as f:
    pickle.dump((samples_raw, bounces, z_true, beta_true, runtime), f)
