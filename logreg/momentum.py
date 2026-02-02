import numpy as np


class Gaussian():

    @staticmethod
    def draw(n=1):
        return np.random.normal(size=n)

    @staticmethod
    def logdensity(p):
        return np.sum(- 0.5 * p ** 2)

    @staticmethod
    def grad_logdensity(p):
        return -p

    @staticmethod
    def integrate(x, p, num_steps, target_grad_logdensity, step):
        half_step = step / 2
        p = p + half_step * target_grad_logdensity(x)
        for _ in range(num_steps - 1):
            x += step * p
            p = p + step * target_grad_logdensity(x)
        x += step * p
        p = p + half_step * target_grad_logdensity(x)
        return x, p


class Laplace():

    @staticmethod
    def draw(n=1):
        return np.random.laplace(size=n)

    @staticmethod
    def logdensity(p):
        return np.sum(-np.abs(p))

    @staticmethod
    def grad_logdensity(p):
        return np.sign(p)

    @staticmethod
    def integrate(x, p, num_steps, target_grad_logdensity, step):
        half_step = step / 2
        psign = np.sign(p)
        x = x + half_step * psign
        for _ in range(num_steps - 1):
            p_new = p + step * target_grad_logdensity(x)
            same_sign = (np.sign(p_new) == psign)
            p = np.where(same_sign, p_new, -p)
            psign = np.sign(p)
            x = x + step * psign
        p_new = p + step * target_grad_logdensity(x)
        same_sign = (np.sign(p_new) == psign)
        p = np.where(same_sign, p_new, -p)
        psign = np.sign(p)
        x = x + half_step * psign
        return x, p
