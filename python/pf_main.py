"""Simplified Python implementation of the particle filter example."""
from __future__ import annotations

import numpy as np

from .pf_utils import load_dem, dem_height, gaussian_likelihood, systematic_resample


class ParticleFilter:
    """Basic particle filter for terrain-referenced navigation."""

    def __init__(self, dem: dict, num_particles: int = 500, dt: float = 1.0, sig_proc: float = 8.0, sig_meas: float = 25.0):
        self.dem = dem
        self.num_particles = num_particles
        self.dt = dt
        self.sig_proc = sig_proc
        self.sig_meas = sig_meas

        self.particles = np.zeros((2, num_particles))
        self.weights = np.ones(num_particles) / num_particles

    def initialize(self, init_mean: np.ndarray, init_std: float) -> None:
        self.particles[:] = init_mean[:, None] + init_std * np.random.randn(2, self.num_particles)
        self.weights[:] = np.ones(self.num_particles) / self.num_particles

    def predict(self, v: np.ndarray) -> None:
        noise = self.sig_proc * np.random.randn(2, self.num_particles) * self.dt
        self.particles += (v[:, None] * self.dt) + noise

    def update(self, z: float) -> None:
        for i in range(self.num_particles):
            z_est = dem_height(self.particles[:, i], self.dem)
            self.weights[i] *= gaussian_likelihood(z_est, z, self.sig_meas)
        self.weights /= np.sum(self.weights)
        self.particles, self.weights = systematic_resample(self.particles, self.weights)

    def estimate(self) -> np.ndarray:
        return np.mean(self.particles, axis=1)


def run_example(dem_path: str) -> None:
    """Run the particle filter with synthetic data similar to the MATLAB demo."""
    dem = load_dem(dem_path)
    dt = 1.0
    sim_time = 150
    num_steps = int(sim_time / dt) + 1
    time = np.arange(0, sim_time + dt, dt)

    x_init = np.array([2600.0, 2800.0])
    sig_init = 40.0
    sig_v = 2.0
    sig_z = 15 + 4.71

    # generate simple trajectory
    true_pos = np.zeros((2, num_steps))
    true_pos[:, 0] = x_init
    true_vel = np.zeros((2, num_steps - 1))
    true_vel[:, :] = np.array([[42.0], [0.0]])
    for k in range(1, num_steps):
        true_pos[:, k] = true_pos[:, k - 1] + true_vel[:, k - 1] * dt

    # noisy measurements
    rng = np.random.default_rng()
    z_meas = np.array([dem_height(true_pos[:, k], dem) + rng.normal(scale=sig_z) for k in range(num_steps)])
    v_meas = true_vel + rng.normal(scale=sig_v, size=true_vel.shape)

    pf = ParticleFilter(dem)
    pf.initialize(x_init, sig_init)

    est = np.zeros((2, num_steps))
    est[:, 0] = pf.estimate()

    for k in range(1, num_steps):
        pf.predict(v_meas[:, k - 1])
        pf.update(z_meas[k])
        est[:, k] = pf.estimate()

    # simple output: RMS error
    err = np.linalg.norm(est - true_pos, axis=0)
    print(f"RMS error: {np.sqrt(np.mean(err ** 2)):.2f} m")


if __name__ == "__main__":
    run_example("DB_part.mat")
