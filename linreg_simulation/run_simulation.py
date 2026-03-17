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
import cProfile
import pickle
import pstats
import numpy as np
import os

sys.path.append('.')

from linreg.sampler import sample

method = sys.argv[1]
n_iter = int(sys.argv[2])
thin = int(sys.argv[3])
seed = int(sys.argv[4])
p_slab = float(sys.argv[5])
alpha = float(sys.argv[6])
if method != "shzz":
    thin = 1  # PDMPs just use longer discretization times
t1 = float(sys.argv[7])
try:
    t = [t1, float(sys.argv[8])]
    t_str = "-".join([str(i).replace(".", "-") for i in t])
except IndexError:
    t = t1
    t_str = str(t).replace(".", "-")
node = socket.gethostname().split(".")[0]

with open(f"linreg_simulation/simulated_data_a{str(alpha).replace('.', '-')}.p", "rb") as f:
    X, y, z_true, beta_true, sigma2 = pickle.load(f)

tau2 = 1
initial_perturbed = beta_true * z_true + np.random.normal(size=len(beta_true), scale=0.001)
profiler = cProfile.Profile()
profiler.enable()
samples_raw, aug_samples_raw, bounces, sampler, runtime = sample(
    method, n_iter, initial_perturbed, z_true, X, y, sigma2, p_slab, tau2, t, thin, seed=seed)


filename = (f'{method}_'
            f'seed{seed}_'
            f't{t_str}_'
            f'thin{thin}_'
            f'pslab{str(p_slab).replace(".", "-")}_'
            f'tau2{str(tau2).replace(".", "-")}_'
            f'iter{n_iter}_'
            f'alpha{str(alpha).replace(".", "-")}_'
            f'{node}')
profiler.disable()
pstats.Stats(profiler).dump_stats(
    f"{filename}.prof",
)
with open(f"{filename}.p", "wb") as f:
    pickle.dump((samples_raw, bounces, z_true, beta_true, runtime), f)  # not storing aug_samples for space
