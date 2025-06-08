"""Utility functions for particle filtering in Python."""
from __future__ import annotations

import numpy as np
try:
    from scipy.io import loadmat
    from scipy.interpolate import RectBivariateSpline
except Exception:  # pragma: no cover - SciPy may not be available
    loadmat = None
    RectBivariateSpline = None


def load_dem(path: str) -> dict:
    """Load DEM data from a .mat file returned as a dictionary."""
    if loadmat is None:
        raise ImportError("SciPy is required to load MATLAB files")
    data = loadmat(path)
    if 'DB' not in data or 'resolution' not in data:
        raise KeyError("MAT file does not contain expected fields")
    return {
        'DB': data['DB'],
        'resolution': float(data['resolution'].squeeze())
    }


def dem_height(pos: np.ndarray, dem: dict) -> float:
    """Interpolate the DEM height at a given x, y position."""
    if RectBivariateSpline is None:
        raise ImportError("SciPy is required for interpolation")
    x, y = pos
    resolution = dem['resolution']
    grid = dem['DB']
    r, c = grid.shape
    ix = int(np.floor(x / resolution))
    iy = int(np.floor(y / resolution))
    ix_idx = np.arange(ix - 2, ix + 3)
    iy_idx = np.arange(iy - 2, iy + 3)
    ix_idx = np.clip(ix_idx, 0, r - 1)
    iy_idx = np.clip(iy_idx, 0, c - 1)
    patch = grid[np.ix_(ix_idx, iy_idx)]
    X = ix_idx * 1.0
    Y = iy_idx * 1.0
    interp = RectBivariateSpline(X, Y, patch, kx=3, ky=3)
    return float(interp(x / resolution, y / resolution))


def gaussian_likelihood(z_est: float, z_mea: float, sig: float) -> float:
    """Gaussian likelihood of z_mea given estimate z_est."""
    var = sig ** 2
    return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-((z_mea - z_est) ** 2) / (2.0 * var))


def systematic_resample(particles: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Systematic resampling for particle filters."""
    weights = weights / np.sum(weights)
    cdf = np.cumsum(weights)
    num = len(weights)
    indexes = np.searchsorted(cdf, np.random.rand(num))
    resampled = particles[:, indexes]
    return resampled, np.ones(num) / num
