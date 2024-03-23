import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import multivariate_normal
from typing import Tuple


np.random.seed(0)


class Agent:
    def __init__(self, init_state: np.ndarray, dt: float, init_uncertainty=1e-1, process_noise=1e-2, measurement_noise=1e-2):
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


def simulate(path: np.ndarray, dt=0.01):
    timestamps = np.linspace(0, len(path) * dt, len(path))
    vels = np.diff(path, axis=0) / dt
    init_state = [*path[0], *vels[0]]
    agent = Agent(init_state, dt=dt)
    positions = np.zeros((len(path), 2))
    positions[0] = path[0]
    covariances = np.zeros((len(path), 2, 2))
    covariances[0] = agent.get_pos_cov()

    for i in range(1, len(path)):
        agent.predict()
        positions[i] = agent.x[:2]
        covariances[i] = agent.get_pos_cov()
        agent.update(vels[i-1])
    return timestamps, positions, covariances



def generate_path(n_steps, domain_dim, smoothing_window=200):
    """Generate a random continuous path in 2D space"""
    path = np.cumsum(np.random.randn(n_steps, domain_dim), axis=0)
    smoothing_window = min(smoothing_window, n_steps)
    window = np.hanning(smoothing_window)
    path[:, 0] = np.convolve(path[:, 0], window, mode='same')
    path[:, 1] = np.convolve(path[:, 1], window, mode='same')
    path = 2 * path / np.max(np.abs(path), axis=0)
    path -= path[0]
    return path


def plot_bounded_path(ts: np.ndarray, *paths: Tuple[np.ndarray, np.ndarray]):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    dims = ['x', 'y']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    bound_params = dict(line=dict(dash='dash'), showlegend=False, mode='lines', col=1)

    for idx, (path, covariances) in enumerate(paths):
        design_params = dict(mode='lines', showlegend=len(paths) > 1, col=1)
        design_params['marker'] = dict(color=colors[idx])
        bound_params['marker'] = dict(color=colors[idx])
        for dim, dim_label in enumerate(dims):
            design_params['row'] = dim + 1
            bound_params['row'] = dim + 1
            stds = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))    
            fig.add_scatter(x=ts, y=path[:,dim], name=f"Path {idx}", **design_params)
            fig.add_scatter(x=ts, y=path[:,dim] + stds[:,dim], **bound_params)
            fig.add_scatter(x=ts, y=path[:,dim] - stds[:,dim], **bound_params)
            design_params['showlegend'] = False

    for i, dim_label in enumerate(dims, 1):
        fig.update_yaxes(title_text=f"{dim_label.capitalize()}", row=i, col=1)
    fig.update_xaxes(title_text="Time (s)", row=len(dims), col=1)
    fig.show()


def get_map_space(positions: np.ndarray, ppm=1):
    """Returns 2x2 array of maps"""
    max_dims = np.max(positions, axis=0) + 1
    min_dims = np.min(positions, axis=0)
    points = (max_dims - min_dims) * ppm
    points = points.astype(int)
    xs = np.linspace(min_dims[0], max_dims[0], points[0])
    ys = np.linspace(min_dims[1], max_dims[1], points[1])
    return xs, ys

def gaussian_2d(xs, ys, mean, cov):
    mean = np.array([0, 0])
    cov = np.array([[1, 0], [0, 1]])
    rv = multivariate_normal(mean, cov)
    return np.array([[rv.pdf([x, y]) for x in xs] for y in ys])

def animate(positions: np.ndarray, covariances: np.ndarray, ppm: int=10):
    xs, ys = get_map_space(positions, ppm)
    dists = np.array([gaussian_2d(xs, ys, pos, cov) for pos, cov in zip(positions, covariances)])
    print(dists.shape)
    max_amp = np.max(dists, axis=None)
    heatmaps = [go.Heatmap(x=xs, y=ys, z=dist, zmax=max_amp, colorscale='Viridis') for dist in dists]
    frames = [go.Frame(data=[heatmap]) for heatmap in heatmaps]
    fig = go.Figure(
        data = [heatmaps[0]],
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                            method="animate",
                            args=[None])])],
            title="Heatmap Animation",
            xaxis_title="X Position",
            yaxis_title="Y Position"
        ),
        frames=frames,
    )
    fig.show()


if __name__ == "__main__":
    path = generate_path(1000, 2)
    timestamps, positions, covariances = simulate(path)
    plot_bounded_path(timestamps, [positions, covariances])
