"""Functions for simulating motion of SHBPS, agnostic of model."""

import numpy as np
import scipy as sp

from linreg.particle.idxs import min_argmin


def compute_freeze_boundary(p_slab, tau2):
    """Compute freeze boundary for deterministic dynamics."""
    kappa = p_slab / (1 - p_slab) / np.sqrt(2 * np.pi * tau2)
    return 1 / kappa / 2


def reflect(v: np.ndarray,
            ortho_plane: np.ndarray):
    """Reflect the vector p off of the surface orthogonal to ortho_plane."""
    return v - 2 * (ortho_plane @ v) / (ortho_plane @ ortho_plane) * ortho_plane


def time_to_unthaw(x: np.ndarray,
                   v: np.ndarray,
                   freeze_boundaries: np.ndarray,
                   positive_threshold: float = 1e-12):
    """
    Find first time and index to unthaw by exiting augmented space.

    Assumes boundaries are equal on both sides of zero, i.e. freeze_bounaries < x < freeze_boundaries.
    """
    next_unthaw = (np.sign(v) * freeze_boundaries - x) / v
    next_unthaw[next_unthaw < positive_threshold] = np.inf
    return min_argmin(next_unthaw)
