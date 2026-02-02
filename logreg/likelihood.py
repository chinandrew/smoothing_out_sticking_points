import numpy as np
from scipy.special import expit


def loglik(beta, X, y):
    z = X @ beta
    return np.sum(y * z - np.log1p(np.exp(z)))


def loglik_Xb(Xbeta, y):
    return np.sum(y * Xbeta - np.log1p(np.exp(Xbeta)))


def grad_loglike(beta, X, y):
    p = expit(X @ beta)
    return X.T @ (y - p)


def logprior(beta, tau2):
    return - beta @ (beta / (2 * tau2))


def grad_logprior(beta, tau2):
    return - beta / tau2


def logdensity(beta, X, y, tau2):
    return loglik(beta, X, y) + logprior(beta, tau2)


def grad_logdensity(beta, X, y, tau2):
    return grad_loglike(beta, X, y) + grad_logprior(beta, tau2)
