import numpy as np
from logging import getLogger

import pdf
from utils import plot_heatmaps, plot_bounded_path, get_sample_spacing, generate_path


class Agent:
    def __init__(self, init_state: np.ndarray, dt: float, init_uncertainty=1e-5, process_noise=1e-4, measurement_noise=1e-4):
        if len(init_state) != 4:
            raise ValueError("Initial state must be a 4D vector [x, y, vx, vy]")
        self.x = np.array(init_state)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.P = np.eye(4) * init_uncertainty # Initial covariance matrix
        self.Q = np.eye(4) * process_noise # Process noise
        self.R = np.eye(2) * measurement_noise # Measurement noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = self.P - K @ self.H @ self.P

    def get_pos_cov(self):
        return self.P[:2, :2]


def simulate(path: np.ndarray, dt=0.01, noise=1e-5):
    log = getLogger(__name__)
    log.info("Starting Kalman simulation...")
    timestamps = np.linspace(0, len(path) * dt, len(path))
    vels = np.diff(path, axis=0) / dt
    if noise:
        vels += np.random.randn(*vels.shape) * noise
    init_state = [*path[0], *vels[0]]
    agent = Agent(init_state, dt=dt, measurement_noise=noise)
    positions = np.zeros((len(path), 2))
    positions[0] = path[0]
    covariances = np.zeros((len(path), 2, 2))
    covariances[0] = agent.get_pos_cov()

    for i in range(1, len(path)):
        agent.predict()
        positions[i] = agent.x[:2]
        covariances[i] = agent.get_pos_cov()
        agent.update(vels[i-1])
    log.info("Kalman simulation complete")
    return timestamps, positions, covariances


def get_map_space(positions: np.ndarray, ppm=1, padding=0.1):
    """Returns 2x2 array of maps"""
    max_dims = np.max(positions, axis=0) + 1
    min_dims = np.min(positions, axis=0)
    delta = max_dims - min_dims
    delta += delta * padding
    points = (delta * ppm).astype(int)
    xs = np.linspace(min_dims[0], max_dims[0], points[0])
    ys = np.linspace(min_dims[1], max_dims[1], points[1])
    return xs, ys


def plot_kalman_heatmaps(positions: np.ndarray, covariances: np.ndarray, bounds=None, ppm=30, num_plots=9, **kwargs):
    t_spacing = get_sample_spacing(positions, num_plots)
    p = positions[::t_spacing]
    c = covariances[::t_spacing]
    if bounds is None:
        xs, ys = get_map_space(positions, ppm)
    else:
        delta = bounds[1] - bounds[0]
        xs, ys = np.linspace(bounds[0], bounds[1], delta*ppm), np.linspace(bounds[0], bounds[1], delta*ppm)
    dists = np.array([pdf.gaussian2d(xs, ys, pos, cov) for pos, cov in zip(p, c)])
    zmax = np.max(dists, axis=None)
    dists /= zmax
    plot_heatmaps(xs, ys, dists, num_plots, **kwargs)


if __name__ == "__main__":
    np.random.seed(0)
    path = generate_path(1000, 2)
    timestamps, positions, covariances = simulate(path, noise=1e-1)
    stds = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
    # plot_bounded_path(timestamps, [path, np.zeros((len(path), 2))], [positions, stds])
    plot_kalman_heatmaps(positions, covariances)
