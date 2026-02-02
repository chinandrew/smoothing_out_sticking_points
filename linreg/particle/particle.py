"""Sample linear regression model using sticky HBPS sampler."""
from abc import ABC, abstractmethod

import numpy as np

from linreg.particle.dynamics import compute_freeze_boundary
from linreg.particle.idxs import convert_to_full_idx, min_argmin, convert_to_sub_idx


class Particle(ABC):
    """Class to store state and simulate SHBPS dynamics."""

    freeze_boundaries: np.ndarray

    def __init__(self,
                 x: np.ndarray,
                 unfrozen: np.ndarray,
                 v: np.ndarray,
                 design: np.ndarray,
                 response: np.ndarray,
                 sigma2: float,
                 tau2: float,
                 p_slab: float,
                 init_frozen_random: bool):
        """TODO."""
        self.d = len(x)
        self.x = np.copy(x)
        self.v = np.copy(v)
        self.unfrozen = np.copy(unfrozen)
        self.prec = 1 / tau2 * np.eye(len(x)) + design.T @ design / sigma2
        self.XTys2 = design.T @ response / sigma2
        self.sigma2 = sigma2
        self.tau2 = tau2
        self.num_unfrozen = np.sum(self.unfrozen)
        self.precomp = {}
        self.freeze_boundaries = np.ones(self.d) * compute_freeze_boundary(p_slab, tau2)
        self.info = {"num_bounces": 0, "num_freeze_thaw": 0, "num_refresh": 0}
        if not init_frozen_random:
            if any(~self.unfrozen) and not all(abs(self.x[~self.unfrozen]) < self.freeze_boundaries[~self.unfrozen]):
                raise ValueError("Starting location for frozen coordinates outside of freeze boundary.")
            if any(~self.unfrozen) and len(set(self.x[~self.unfrozen])) < len(self.x[~self.unfrozen]):
                raise ValueError("Starting location for multiple frozen coordinates cannot be the same.")
        else:
            self.init_frozen_randomly()
        self.freeze_times = np.full_like(x, np.inf)
        self.freeze_times[~self.unfrozen] = (np.sign(self.v[~self.unfrozen] ) * self.freeze_boundaries[~self.unfrozen]  - self.x[~self.unfrozen] ) / self.v[~self.unfrozen]

    @property
    def true_x(self):
        """Return true position, remapping all frozen elements to 0."""
        return self.x * self.unfrozen

    def update_frozen_x(self):
        self.x[~self.unfrozen] = np.sign(self.v[~self.unfrozen]) * (self.freeze_boundaries[~self.unfrozen] - self.freeze_times[~self.unfrozen])

    @property
    def aug_x(self):
        """Augmented position."""
        self.update_frozen_x()
        augmented_position = np.copy(self.x)
        negatives = ~self.unfrozen & (self.x < 0)
        positives = ~self.unfrozen & (self.x > 0)
        if any(negatives):
            augmented_position[negatives] -= self.freeze_boundaries[negatives]
        if any(positives):
            augmented_position[positives] += self.freeze_boundaries[positives]
        return augmented_position

    def advance_frozen(self, t: float):
        """Advance dynamics for frozen coordinates."""
        self.freeze_times -= t

    def advance(self, t: float):
        """Advance dynamics for specified time."""
        remaining_time = t
        while remaining_time > 0:
            next_event_time, next_event_idx, next_event_type = self.find_next_event()
            if next_event_time > remaining_time:
                self.advance_both_universes(remaining_time)
                self.update_prec_matvecs(remaining_time, np.nan, None)
                return
            if next_event_type == "bounce":
                self.advance_both_universes(next_event_time)
                self.bounce(next_event_idx)
                self.info["num_bounces"] += 1
            elif next_event_type in ["freeze", "thaw"]:
                self.advance_both_universes(next_event_time)
                self.transition_universe(next_event_idx)
                self.info["num_freeze_thaw"] += 1
                if next_event_type == "freeze":
                    self.num_unfrozen -= 1
                else:
                    self.num_unfrozen += 1
            elif next_event_type == "refresh":
                self.advance_both_universes(next_event_time)
                self.refresh()
                self.info["num_refresh"] += 1
            else:
                raise NotImplementedError
            self.update_prec_matvecs(next_event_time, next_event_idx, next_event_type)
            remaining_time -= next_event_time

    def advance_both_universes(self, t: float):
        """
        Advance dynamics, ignoring any events, for set amount of time.

        Because events are ignored, `t` should only ever ber computed by `find_next_event()`
        """
        if self.num_unfrozen < self.d:
            self.advance_frozen(t)
        if self.num_unfrozen:
            self.advance_unfrozen(t)

    def transition_universe(self, next_event_idx: int):
        """Move specific index in or out of frozen universe. Assumes only index each event."""
        if ~self.unfrozen[next_event_idx]:
            self.x[next_event_idx] = np.nextafter(0, np.sign(self.v[next_event_idx]))
            self.freeze_times[next_event_idx] = np.inf
        else:
            self.freeze_times[next_event_idx] = self.freeze_boundaries[next_event_idx]*2
        self.unfrozen[next_event_idx] = ~self.unfrozen[next_event_idx]

    @abstractmethod
    def advance_unfrozen(self, t: float):
        """Advance dynamics for unfrozen coordinates."""

    def find_next_event(self):
        """Find next event of any kind."""
        next_frozen_event_time, next_frozen_event_idx, next_frozen_event_type = self.next_frozen_event()
        if not self.num_unfrozen:
            return next_frozen_event_time, next_frozen_event_idx, next_frozen_event_type
        next_standard_event_time, next_standard_event_idx, next_standard_event_type = self.next_unfrozen_event()
        if next_frozen_event_time < next_standard_event_time:
            return next_frozen_event_time, next_frozen_event_idx, next_frozen_event_type
        return next_standard_event_time, next_standard_event_idx, next_standard_event_type

    @abstractmethod
    def time_to_bounce(self):
        """Compute next time to bounce, ignoring freezes."""

    def time_to_zero(self, positive_threshold: float = 1e-12):
        """Find first time and index to reach 0."""
        next_zeros = -self.x[self.unfrozen] / self.v[self.unfrozen]
        next_zeros[next_zeros < positive_threshold] = np.inf
        return min_argmin(next_zeros)

    @abstractmethod
    def bounce(self, bounce_idx):
        """Execute bounce event."""

    def next_unfrozen_event(self):
        """Compute next time an event occurs in the unfrozen universe, either a bounce or a freeze."""
        next_bounce_time, next_bounce_idx = self.time_to_bounce()
        next_zero_time, next_zero_idx = self.time_to_zero()
        if next_bounce_time < next_zero_time:
            return next_bounce_time, convert_to_full_idx(self.unfrozen, next_bounce_idx), "bounce"
        return next_zero_time, convert_to_full_idx(self.unfrozen, next_zero_idx), "freeze"

    def next_frozen_event(self):
        """Compute next time an event occurs in the frozen universe. Currently the only such event is unfreezing."""
        next_thaw_time, next_thaw_idx = min_argmin(self.freeze_times)
        return next_thaw_time, next_thaw_idx, "thaw"

    def cache(self, next_event_time, next_event_idx, next_event_type, remaining_time):
        """Store any information about final state if needed."""

    def grad_U(self):
        """Compute current gradient of U."""
        return self.prec[self.unfrozen, :][:, self.unfrozen] @ self.x[self.unfrozen] - self.XTys2[self.unfrozen]

    def prec_matvecs(self):
        """Compute precision matrix times position and velocity vectors"""
        prec = self.prec[self.unfrozen, :][:, self.unfrozen]
        precv = prec @ self.v[self.unfrozen]
        precx = prec @ self.x[self.unfrozen]
        precx_mu = precx - self.XTys2[self.unfrozen]
        return precv, precx, precx_mu

    def update_prec_matvecs_bounce(self, next_event_idx):
        self.precomp["precv"] =  self.precomp["precv"] + 2 * self.v[next_event_idx] * self.prec[self.unfrozen, next_event_idx]

    def update_prec_matvecs_freeze(self, next_event_idx):
        previous_unfrozen_state = np.copy(self.unfrozen)
        previous_unfrozen_state[next_event_idx] = ~previous_unfrozen_state[next_event_idx]
        idx_to_delete = convert_to_sub_idx(previous_unfrozen_state, next_event_idx)
        mask = np.full(len(self.precomp["precv"]), True)  # faster than using np.delete
        mask[idx_to_delete] = False
        self.precomp["precv"] = self.precomp["precv"][mask] - self.prec[self.unfrozen, next_event_idx] * self.v[next_event_idx]
        self.precomp["precx"] = self.precomp["precx"][mask] - self.prec[self.unfrozen, next_event_idx] * self.true_x[next_event_idx]

    def update_prec_matvecs_thaw(self, next_event_idx):
        previous_unfrozen_state = np.copy(self.unfrozen)
        previous_unfrozen_state[next_event_idx] = ~previous_unfrozen_state[next_event_idx]
        prec12 = prec21 = self.prec[previous_unfrozen_state, next_event_idx]
        prec22 = self.prec[next_event_idx, next_event_idx]
        v1 = self.v[previous_unfrozen_state]
        v2 = self.v[next_event_idx]
        x1 = self.x[previous_unfrozen_state]
        # x2 = self.x[next_event_idx]
        idx_to_insert = convert_to_sub_idx(self.unfrozen, next_event_idx)
        new_len = len(self.precomp["precv"])+1
        mask = np.full(new_len, True)  # faster than using np.insert
        mask[idx_to_insert] = False
        new_precx = np.empty(new_len)
        new_precx[mask] = self.precomp["precx"]
        new_precx[idx_to_insert] = prec21 @ x1
        self.precomp["precx"] = new_precx
        new_precv = np.empty(new_len)
        new_precv[mask] = self.precomp["precv"] + prec12 * v2
        new_precv[idx_to_insert] = prec21 @ v1 + prec22 * v2
        self.precomp["precv"] = new_precv

    def update_prec_matvecs(self, next_event_time: float, next_event_idx: int, next_event_type: str):
        self.precomp["precx"] = self.precomp["precx"] + next_event_time * self.precomp["precv"]
        if next_event_type == "bounce":
            self.update_prec_matvecs_bounce(next_event_idx)
        elif next_event_type == "freeze":
            self.update_prec_matvecs_freeze(next_event_idx)
        elif next_event_type == "thaw":
            self.update_prec_matvecs_thaw(next_event_idx)
        else:
            pass
        self.precomp["precx_mu"] = self.precomp["precx"] - self.XTys2[self.unfrozen]
        #assert np.allclose(self.precomp["precv"], self.prec[self.unfrozen, :][:, self.unfrozen] @ self.v[self.unfrozen])
        #assert np.allclose(self.precomp["precx"], self.prec[self.unfrozen, :][:, self.unfrozen] @ self.x[self.unfrozen])
        return

    def refresh(self):
        """Refresh any necessary parameters"""
        pass

    def init_frozen_randomly(self):
        self.x[~self.unfrozen] = np.random.uniform(-0.99 * self.freeze_boundaries[~self.unfrozen],
                                                   0.99 * self.freeze_boundaries[~self.unfrozen])