"""Samplers for Bayesian spike and slab linear regression.

Assumes y ~ N(X@beta, sigma2), beta ~ N(0, tau2), and z ~ Bernoulli(p_slab)

"""
import time
from typing import Union

import numpy as np

from linreg.particle.shzz_particle import SHZZParticle
from linreg.particle.szz_particle import SZZParticle


def sample(method, n_iter, initial, z_true, X, y, sigma2, p_slab, tau2, t, thin=1, seed=0):
    np.random.seed(seed)
    initial = np.copy(initial)
    z = np.copy(z_true)
    start = time.time()
    if method == "szz":
        samples, aug_samples, bounces, sampler = szz(n_iter, z.astype(bool), initial, X, y, sigma2, p_slab, tau2,
                                                     travel_time=t, init_unif_frozen=False)
    elif method == "szz-constant":
        samples, aug_samples, bounces, sampler = szz(n_iter, z.astype(bool), initial, X, y, sigma2, p_slab, tau2,
                                                     travel_time=t, random_freeze_time=False, init_unif_frozen=True)
    elif method == "shzz":
        samples, aug_samples, bounces, sampler = shzz(n_iter, z.astype(bool), initial, X, y, sigma2, p_slab, tau2,
                                                      travel_time=t, thin=thin, init_unif_frozen=True)
    else:
        raise NotImplementedError
    runtime = time.time() - start
    return samples, aug_samples, bounces, sampler, runtime


def shzz(n_iters: int,
         unfrozen: np.ndarray,
         beta: np.ndarray,
         X: np.ndarray,
         y: np.ndarray,
         sigma2: float,
         p_slab: float,
         tau2: float,
         travel_time: Union[np.ndarray, list, float] = 1,
         thin: int = 1,
         init_unif_frozen: bool = False):
    """
    Draw posterior samples using Sticky HZZ.

    :param n_iters: Number of samples to draw.
    :param unfrozen: Initial boolean vector for which of the `beta` are unfrozen. Length p.
    :param beta: Initial vector for latent coefficients beta. If unfrozen[i] == False, then beta[i] is
        its position in the augmented universe. Length p.
    :param X: Design matrix. Dimension n x p.
    :param y: Response vector. Length p.
    :param sigma2: Variance of y.
    :param p_slab: Prior probability of being 1 for z.
    :param tau2: Prior variance for beta.
    :param travel_time: Time SHZZ dynamics are simulated for each sample. If a length 2 list,
        denotes lower/upper bound of uniform draw for travel times.
    :param thin: Number of samples to thin.
    :param init_unif_frozen: Whether or not to initialize frozen coordinates uniformily within
        -0.99*freeze_boundaries to 0.99*freeze_boundaries.
    :return: n_iter x p matrix of samples of beta
    """
    p = X.shape[1]
    samples = np.empty((n_iters, p))
    aug_samples = np.empty((n_iters, p))
    bounces = np.empty((n_iters, 2))
    momentum = np.random.laplace(size=p)
    sampler = SHZZParticle(beta, unfrozen, momentum, p_slab, X, y, sigma2, tau2, init_unif_frozen)
    for i in range(int(n_iters * thin)):
        if isinstance(travel_time, (int, float)):
            sampler.advance(travel_time)
        else:
            sampler.advance(np.random.uniform(travel_time[0], travel_time[1]))
        if thin == 1 or i % thin == (thin - 1):
            samples[int(i / thin), :] = sampler.true_x
            aug_samples[int(i / thin), :] = sampler.aug_x
            bounces[int(i / thin), 0] = sampler.info["num_bounces"]
            bounces[int(i / thin), 1] = sampler.info["num_freeze_thaw"]
        sampler.refresh()
    return samples, aug_samples, bounces, sampler


def szz(n_iters: int,
        unfrozen: np.ndarray,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        sigma2: float,
        p_slab: float,
        tau2: float,
        travel_time: float = 1,
        random_freeze_time: bool = True,
        init_unif_frozen: bool = False):
    """
    Draw posterior samples using Sticky ZZ.

    :param n_iters: Number of samples to draw.
    :param unfrozen: Initial boolean vector for which of the `beta` are unfrozen. Length p.
    :param beta: Initial vector for latent coefficients beta. If unfrozen[i] == False, then beta[i] is
        its position in the augmented universe. Length p.
    :param X: Design matrix. Dimension n x p.
    :param y: Response vector. Length p.
    :param sigma2: Variance of y.
    :param p_slab: Prior probability of being 1 for z.
    :param tau2: Prior variance for beta.
    :param travel_time: Discretization time. How often samples are captured from continuous trajectory.
    :param random_freeze_time: Whether the amount of time frozen should be random or fixed.
    :param init_unif_frozen: Whether or not to initialize frozen coordinates uniformily within
        -0.99*freeze_boundaries to 0.99*freeze_boundaries.
    :return: n_iter x p matrix of samples of beta
    """
    p = X.shape[1]
    samples = np.empty((n_iters, p))
    aug_samples = np.empty((n_iters, p))  if not random_freeze_time else None
    bounces = np.empty((n_iters, 2))
    v = np.random.binomial(1, 0.5, size=p) * 2 - 1
    sampler = SZZParticle(beta, unfrozen, v, p_slab, X, y, sigma2, tau2, random_freeze_time, init_unif_frozen)
    for i in range(n_iters):
        sampler.advance(travel_time)
        samples[i, :] = sampler.true_x
        if not random_freeze_time:
            aug_samples[i, :] = sampler.aug_x
        bounces[i, 0] = sampler.info["num_bounces"]
        bounces[i, 1] = sampler.info["num_freeze_thaw"]
    return samples, aug_samples, bounces, sampler
