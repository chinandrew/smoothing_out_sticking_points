import numpy as np
from numba import njit

from likelihood import logdensity, grad_logdensity
from momentum import Laplace


@njit(cache=True)
def expit(z):
    """Since used in another numba function"""
    return 1 / (1 + np.exp(-z))


def sample_hmc(x, p, target_logdensity, target_grad_logdensity, momentum, num_steps, step):
    x_proposal = np.copy(x)
    start_logdensity = target_logdensity(x) + momentum.logdensity(p)
    x_proposal, p_proposal = momentum.integrate(x_proposal, p, num_steps, target_grad_logdensity, step)
    end_logdensity = target_logdensity(x_proposal) + momentum.logdensity(p_proposal)
    if np.log(np.random.uniform()) < end_logdensity - start_logdensity:
        return x_proposal, p_proposal, True, np.exp(end_logdensity - start_logdensity)
    else:
        return x, p, False, np.exp(end_logdensity - start_logdensity)


def collapse_latent(x, boundary):
    """~3.5x slower than jit version, doesn't make a huge difference overall.

    Could probably further optimize when collapsing happens but haven't gotten around to that.
    """
    output = x - np.sign(x) * boundary
    mask = np.abs(x) > boundary
    output = np.where(mask, output, 0.0)
    return output, mask


def compute_latent_width(p_slab, tau2):
    """Compute freeze boundary for normal prior on beta."""
    widths = np.zeros(len(p_slab))
    has_slab = p_slab != 1
    widths[has_slab] = (1 - p_slab[has_slab]) / (p_slab[has_slab] / np.sqrt(2 * np.pi * tau2[has_slab]))
    return widths


def hmc(iters, init, X, y, tau2, p_slab, logdensity, grad_logdensity, momentum, min_steps, step, thin=1):
    x = np.copy(init)
    boundary = compute_latent_width(p_slab, tau2) / 2

    def collapsed_logdensity(beta):
        collapse_beta, mask = collapse_latent(beta, boundary)
        return logdensity(collapse_beta[mask], X[:, mask], y, tau2[mask])

    def collapsed_grad_logdensity(beta):
        grad = np.zeros(len(beta))
        collapse_beta, mask = collapse_latent(beta, boundary)
        grad[mask] = grad_logdensity(collapse_beta[mask], X[:, mask], y, tau2[mask])
        return grad

    accepts = 0
    num_samples = int(np.ceil(iters / thin))
    samples = {"beta": np.empty((num_samples, len(x))),
               "latent": np.empty((num_samples, len(x))),
               "hamiltonian_diff": np.empty(num_samples)}
    p = momentum.draw(len(x))
    for i in range(iters):
        num_steps = np.random.randint(min_steps, min_steps * 3)
        x, p, accept, hamil_diff = sample_hmc(x,
                                              p,
                                              collapsed_logdensity,
                                              collapsed_grad_logdensity,
                                              momentum,
                                              num_steps,
                                              step * np.random.uniform(0.8, 1.2))
        c, mask = collapse_latent(x, boundary)
        p[mask] = momentum.draw(np.sum(mask))
        accepts += accept
        if i % thin == 0:
            idx = int(i / thin)
            samples["beta"][idx] = c
            samples["latent"][idx] = x
            samples["hamiltonian_diff"][idx] = hamil_diff
    return samples, {"accept_rate": accepts / iters}


def zz(iters, init, X, y, tau2, p_slab, logdensity, grad_logdensity, step, thin, adjust):
    x = np.copy(init)
    boundary = compute_latent_width(p_slab, tau2) / 2

    def collapsed_logdensity(beta):
        collapse_beta, mask = collapse_latent(beta, boundary)
        return logdensity(collapse_beta[mask], X[:, mask], y, tau2[mask])

    def collapsed_grad_logdensity(beta):
        grad = np.zeros(len(beta))
        collapse_beta, mask = collapse_latent(beta, boundary)
        grad[mask] = grad_logdensity(collapse_beta[mask], X[:, mask], y, tau2[mask])
        return grad

    accepts = 0
    num_samples = int(np.ceil(iters / thin))
    samples = {"beta": np.empty((num_samples, len(x))),
               "latent": np.empty((num_samples, len(x)))}
    v = np.random.choice([-1, 1], len(init))
    current_logdensity = collapsed_logdensity(x)
    for i in range(iters):
        x, v, accept, current_logdensity = step_zz(x,
                                                   v,
                                                   collapsed_logdensity,
                                                   collapsed_grad_logdensity,
                                                   step * np.random.uniform(0.8, 1.2),
                                                   current_logdensity,
                                                   adjust)
        if i % thin == 0:
            idx = int(i / thin)
            c, mask = collapse_latent(x, boundary)
            samples["beta"][idx] = c
            samples["latent"][idx] = x
        accepts += accept
    return samples, {"accept_rate": accepts / iters}


def step_zz(x, v, target_logdensity, target_grad_logdensity, step, current_logdensity, adjust):
    half_step = step / 2
    x_mid = x + half_step * v
    U = -target_grad_logdensity(x_mid)
    bounce_rates = np.maximum(0, v * U)
    bounces = - ((np.random.uniform(size=len(x)) < 1 - np.exp(-step * bounce_rates)) * 2 - 1)
    proposal_v = v * bounces
    proposal_x = x_mid + half_step * proposal_v
    if not adjust:
        return proposal_x, proposal_v, True, None
    neg_proposal_bounce_rates = np.maximum(0, -proposal_v * U)
    proposal_logdensity = target_logdensity(proposal_x)
    accept_prob = (
            proposal_logdensity
            - current_logdensity
            + step * np.sum(bounce_rates - neg_proposal_bounce_rates))
    if np.log(np.random.uniform()) < accept_prob:
        x, v = proposal_x, proposal_v
        accept = True
        current_logdensity = proposal_logdensity
    else:
        x, v = x, -v
        accept = False
    return x, v, accept, current_logdensity
