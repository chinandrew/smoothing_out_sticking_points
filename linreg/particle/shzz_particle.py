"""Sample linear regression model using sticky HZZ sampler."""

import numpy as np

from linreg.particle.idxs import min_argmin
from linreg.particle.particle import Particle
from linreg.particle.root import min_root


class SHZZParticle(Particle):
    """Class to store state and simulate SHZZ dynamics."""

    def __init__(self,
                 x: np.ndarray,
                 frozen: np.ndarray,
                 p: np.ndarray,
                 p_slab: float,
                 design: np.ndarray,
                 response: np.ndarray,
                 sigma2: float,
                 tau2: float,
                 init_frozen_random: bool):
        """TODO."""
        self.p = np.copy(p)
        v = np.sign(p)
        super().__init__(x, frozen, v, design, response, sigma2, tau2, p_slab, init_frozen_random)
        self.precomp["precv"], self.precomp["precx"], self.precomp["precx_mu"] = self.prec_matvecs()

    def advance_unfrozen(self, t: float):
        """Advance dynamics for unfrozen coordinates."""
        self.p[self.unfrozen] = self.p[self.unfrozen] - t * self.precomp["precx_mu"] - t ** 2 / 2 * self.precomp["precv"]
        self.x[self.unfrozen] = self.x[self.unfrozen] + self.v[self.unfrozen] * t

    def time_to_bounce(self):
        """Compute next time to bounce, ignoring freezes. Bounce index is returned as nan."""
        bounce_times = min_root(self.precomp["precv"] / 2, self.precomp["precx_mu"], -self.p[self.unfrozen])
        next_bounce_time, next_bounce_idx = min_argmin(bounce_times)
        return next_bounce_time, next_bounce_idx

    def bounce(self, bounce_idx):
        """Reflect velocity at the chosen index."""
        self.p[bounce_idx] = 0  # for numerical stability
        self.v[bounce_idx] *= -1

    def refresh(self):
        """Draw new momentum and velocity."""
        self.p[self.unfrozen] = np.random.laplace(size=sum(self.unfrozen))
        self.v = np.sign(self.p)
        self.precomp["precv"], self.precomp["precx"], self.precomp["precx_mu"] = self.prec_matvecs()
