"""Sample linear regression model using sticky HBPS sampler."""
from abc import ABC, abstractmethod

import numpy as np

from linreg.particle.dynamics import time_to_unthaw, compute_freeze_boundary
from linreg.particle.idxs import convert_to_full_idx, min_argmin, convert_to_sub_idx


class Particle(ABC):
    """Class to store state and simulate SHBPS dynamics."""

    freeze_boundaries: np.ndarray

    def __init__(self,
                 x: np.ndarray,
                 frozen: np.ndarray,
                 v: np.ndarray,
                 design: np.ndarray,
                 response: np.ndarray,
                 sigma2: float,
                 tau2: float,
                 p_slab):
        """TODO."""
        self.d = len(x)
        self.x = np.copy(x)
        self.v = np.copy(v)
        self.frozen = np.copy(frozen)
        self.prec = 1 / tau2 * np.eye(len(x)) + design.T @ design / sigma2
        self.XTy = design.T @ response
        self.sigma2 = sigma2
        self.tau2 = tau2
        self.precomp = {}
        self.freeze_boundaries = np.ones(self.d) * compute_freeze_boundary(p_slab, tau2)
        self.info = {"num_bounces": 0, "num_freeze_thaw": 0, "num_refresh": 0}
        if any(self.frozen) and not all(abs(self.x[self.frozen]) < self.freeze_boundaries[self.frozen]):
            raise ValueError("Starting location for frozen coordinates outside of freeze boundary.")
        if sum(self.frozen) > 1 and len(set(self.x[self.frozen])) < len(self.x[self.frozen]):
            raise ValueError("Starting location for multiple frozen coordinates cannot be the same.")

    @property
    def true_x(self):
        """Return true position, remapping all frozen elements to 0."""
        return self.x * ~self.frozen

    @property
    def aug_x(self):
        """Augmented position."""
        augmented_position = np.copy(self.x)
        negatives = (~self.frozen) & (self.x < 0)
        positives = (~self.frozen) & (self.x > 0)
        if any(negatives):
            augmented_position[negatives] -= self.freeze_boundaries[negatives]
        if any(positives):
            augmented_position[positives] += self.freeze_boundaries[positives]
        return augmented_position

    def advance_frozen(self, t: float):
        """Advance dynamics for frozen coordinates."""
        self.x[self.frozen] = self.x[self.frozen] + self.v[self.frozen] * t

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
        if any(self.frozen):
            self.advance_frozen(t)
        if any(~self.frozen):
            self.advance_unfrozen(t)

    def transition_universe(self, next_event_idx: int):
        """Move specific index in or out of frozen universe. Assumes only index each event."""
        if self.frozen[next_event_idx]:
            self.x[next_event_idx] = np.nextafter(0, np.sign(self.v[next_event_idx]))
        else:
            self.x[next_event_idx] = -np.sign(self.v[next_event_idx]) * self.freeze_boundaries[next_event_idx]
        self.frozen[next_event_idx] = ~self.frozen[next_event_idx]

    @abstractmethod
    def advance_unfrozen(self, t: float):
        """Advance dynamics for unfrozen coordinates."""

    def find_next_event(self):
        """Find next event of any kind."""
        next_frozen_event_time, next_frozen_event_idx, next_frozen_event_type = self.next_frozen_event()
        if all(self.frozen):
            return next_frozen_event_time, next_frozen_event_idx, next_frozen_event_type
        next_standard_event_time, next_standard_event_idx, next_standard_event_type = self.next_unfrozen_event()
        if next_frozen_event_time < next_standard_event_time:
            return next_frozen_event_time, next_frozen_event_idx, next_frozen_event_type
        return next_standard_event_time, next_standard_event_idx, next_standard_event_type

    @abstractmethod
    def time_to_bounce(self):
        """Compute next time to bounce, ignoring freezes."""

    def time_to_zero(self,  positive_threshold: float = 1e-12):
        """Find first time and index to reach 0."""
        next_zeros = -self.x[~self.frozen] / self.v[~self.frozen]
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
            return next_bounce_time, convert_to_full_idx(~self.frozen, next_bounce_idx), "bounce"
        return next_zero_time, convert_to_full_idx(~self.frozen, next_zero_idx), "freeze"

    def next_frozen_event(self):
        """Compute next time an event occurs in the frozen universe. Currently the only such event is unfreezing."""
        next_thaw_time, next_thaw_idx = time_to_unthaw(self.x[self.frozen], self.v[self.frozen],
                                                       self.freeze_boundaries[self.frozen])
        return next_thaw_time, convert_to_full_idx(self.frozen, next_thaw_idx), "thaw"

    def cache(self, next_event_time, next_event_idx, next_event_type, remaining_time):
        """Store any information about final state if needed."""

    def grad_U(self):
        """Compute current gradient of U."""
        return self.prec[~self.frozen, :][:, ~self.frozen] @ self.x[~self.frozen] - self.XTy[~self.frozen] / self.sigma2

    def prec_matvecs(self):
        """Compute precision matrix times position and velocity vectors"""
        prec = self.prec[~self.frozen, :][:, ~self.frozen]
        precv = prec @ self.v[~self.frozen]
        precx = prec @ self.x[~self.frozen]
        precx_mu = precx - self.XTy[~self.frozen] / self.sigma2
        return precv, precx, precx_mu

    def update_prec_matvecs(self, next_event_time: float, next_event_idx: int, next_event_type: str):
        self.precomp["precx"] = self.precomp["precx"] + next_event_time * self.precomp["precv"]
        if next_event_type == "bounce":
            self.precomp["precv"] = self.precomp["precv"] + 2 * self.v[next_event_idx] * self.prec[
                ~self.frozen, next_event_idx]
        elif next_event_type == "freeze":
            previous_unfrozen_state = np.copy(~self.frozen)
            previous_unfrozen_state[next_event_idx] = ~previous_unfrozen_state[next_event_idx]
            idx_to_delete = convert_to_sub_idx(previous_unfrozen_state, next_event_idx)
            self.precomp["precv"] = np.delete(self.precomp["precv"], idx_to_delete) - self.prec[
                ~self.frozen, next_event_idx] * self.v[next_event_idx]
            self.precomp["precx"] = np.delete(self.precomp["precx"], idx_to_delete) - self.prec[
                ~self.frozen, next_event_idx] * self.true_x[next_event_idx]
        elif next_event_type == "thaw":
            previous_unfrozen_state = np.copy(~self.frozen)
            previous_unfrozen_state[next_event_idx] = ~previous_unfrozen_state[next_event_idx]
            prec12 = prec21 = self.prec[previous_unfrozen_state, next_event_idx]
            prec22 = self.prec[next_event_idx, next_event_idx]
            v1 = self.v[previous_unfrozen_state]
            v2 = self.v[next_event_idx]
            x1 = self.x[previous_unfrozen_state]
            x2 = self.x[next_event_idx]
            idx_to_insert = convert_to_sub_idx(~self.frozen, next_event_idx)
            self.precomp["precx"] = np.insert(self.precomp["precx"] + prec12 * x2, idx_to_insert,
                                              prec21 @ x1 + prec22 * x2)
            self.precomp["precv"] = np.insert(self.precomp["precv"] + prec12 * v2, idx_to_insert,
                                              prec21 @ v1 + prec22 * v2)
        else:
            pass
        self.precomp["precx_mu"] = self.precomp["precx"] - self.XTy[~self.frozen] / self.sigma2
        # assert np.allclose(self.precomp["precv"], self.prec[~self.frozen, :][:, ~self.frozen] @ self.v[~self.frozen])
        # assert np.allclose(self.precomp["precx"], self.prec[~self.frozen, :][:, ~self.frozen] @ self.x[~self.frozen])
        return

    def refresh(self):
        """Refresh any necessary parameters"""
        pass
