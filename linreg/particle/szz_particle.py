"""Sample linear regression model using sticky HZZ sampler."""

import numpy as np

from linreg.particle.dynamics import compute_freeze_boundary
from linreg.particle.idxs import min_argmin
from linreg.particle.particle import Particle
from linreg.particle.root import min_root


class SZZParticle(Particle):
    """Class to store state and simulate SHZZ dynamics."""

    def __init__(self,
                 x: np.ndarray,
                 frozen: np.ndarray,
                 v: np.ndarray,
                 p_slab: float,
                 design: np.ndarray,
                 response: np.ndarray,
                 sigma2: float,
                 tau2: float,
                 random_freeze_time: bool = True):
        """TODO."""
        super().__init__(x, frozen, v, design, response, sigma2, tau2, p_slab)

        self.kappa = 1 / (compute_freeze_boundary(p_slab, tau2) * 2)
        if random_freeze_time:
            self.freeze_boundaries = np.random.exponential(scale=1 / self.kappa, size=self.d) / 2
            self.transition_universe = self.transition_universe_random
        self.exps = np.random.exponential(size=self.d)
        self.precomp["precv"], self.precomp["precx"], self.precomp["precx_mu"] = self.prec_matvecs()

    def advance_unfrozen(self, t: float):
        """Advance dynamics for unfrozen coordinates."""
        self.x[~self.frozen] = self.x[~self.frozen] + self.v[~self.frozen] * t
        change = (
                (t - self.precomp["pos_time"]) * self.precomp["vprecx_mu"]
                + (t ** 2 / 2 - self.precomp["pos_time"] ** 2 / 2) * self.precomp["vprecv"]
        )
        change[self.precomp["pos_time"] > t] = 0
        self.exps[~self.frozen] -= change

    def transition_universe_random(self, next_event_idx: int):
        """Move specific index in or out of frozen universe. Assumes only index each event."""
        if self.frozen[next_event_idx]:
            self.x[next_event_idx] = 0.0
        else:
            self.freeze_boundaries[next_event_idx] = np.random.exponential(scale=1 / self.kappa) / 2
            self.x[next_event_idx] = -np.sign(self.v[next_event_idx]) * self.freeze_boundaries[next_event_idx]
        self.frozen[next_event_idx] = ~self.frozen[next_event_idx]

    def time_to_bounce(self):
        """Compute next time to bounce, ignoring freezes. Bounce index is returned as nan."""
        vprecv = self.v[~self.frozen] * self.precomp["precv"]
        vprecx_mu = self.v[~self.frozen] * self.precomp["precx_mu"]
        pos_time = self.get_first_pos_time(self.precomp["precv"], self.precomp["precx_mu"], vprecv, vprecx_mu)
        c = (- self.exps[~self.frozen] - pos_time * vprecx_mu - pos_time ** 2 / 2 * vprecv)
        bounce_times = min_root(vprecv / 2, vprecx_mu, c, pos_time)
        next_bounce_time, next_bounce_idx = min_argmin(bounce_times)
        self.precomp["vprecv"] = vprecv
        self.precomp["vprecx_mu"] = vprecx_mu
        self.precomp["pos_time"] = pos_time
        return next_bounce_time, next_bounce_idx

    def bounce(self, bounce_idx):
        """Reflect velocity at the chosen index."""
        self.exps = np.random.exponential(size=self.d)
        self.v[bounce_idx] *= -1

    def get_first_pos_time(self, precv, precxmu, vprecv, vprecx_mu):
        """Get time that quadratic is positive for the first time."""
        times = np.full_like(precv, 0)
        vprecv_gt0 = (vprecv >= 0)
        vprecxmu_l0 = (vprecx_mu < 0)
        case1 = vprecxmu_l0 & vprecv_gt0
        times[case1] = - precxmu[case1] / precv[case1]
        times[vprecxmu_l0 & ~vprecv_gt0] = np.inf
        return times
