import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


np.random.seed(0)


class Agent:
    def __init__(self, init_state: np.ndarray, dt: float, init_uncertainty=1e-1, process_noise=1e-2, measurement_noise=1e-1):
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

    def get_std(self):
        return np.diag(self.P)[:2]


def simulate(path: np.ndarray, dt=0.01):
    timestamps = np.linspace(0, len(path) * dt, len(path))
    vels = np.diff(path, axis=0) / dt
    init_state = [*path[0], *vels[0]]
    agent = Agent(init_state, dt=dt)
    positions = np.zeros((len(path), 2))
    positions[0] = path[0]
    confidence = np.zeros((len(path), 2))
    confidence[0] = agent.get_std()

    for i in range(1, len(path)):
        agent.predict()
        positions[i] = agent.x[:2]
        confidence[i] = agent.get_std()
        agent.update(vels[i-1])
    return timestamps, positions, confidence



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


def get_map_space(positions: np.ndarray, ppm=1):
    """Returns 2x2 array of maps"""
    max_dims = np.max(positions, axis=0) + 1
    min_dims = np.min(positions, axis=0)
    points = (max_dims - min_dims) * ppm
    points = points.astype(int)
    xs = np.linspace(min_dims[0], max_dims[0], points[0])
    ys = np.linspace(min_dims[1], max_dims[1], points[1])
    return xs, ys

def gaussian_2d_point(x, y, mu_x, mu_y, std_x, std_y):
    """Returns the value of the 2D Gaussian distribution at (x, y)"""
    term1 = 1 / (2 * np.pi * std_x * std_y)
    term2 = np.exp(-((x - mu_x)**2 / (2 * std_x**2) + (y - mu_y)**2 / (2 * std_y**2)))
    return term1 * term2

def gaussian_2d(xs, ys, mu, std):
    x, y = np.meshgrid(xs, ys)
    return gaussian_2d_point(x, y, *mu, *std)


def compare_paths(ts: np.ndarray, path1: np.ndarray, path2: np.ndarray):
    fig = make_subplots(rows=2, cols=1)
    fig.add_trace(go.Scatter(x=ts, y=path1[:,0], mode='lines', name='Path 1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=path2[:,0], mode='lines', name='Path 2'), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=path1[:,1], mode='lines', name='Path 1'), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=path2[:,1], mode='lines', name='Path 2'), row=2, col=1)
    fig.show()
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=path1[:0], y=path1[:,1], mode='lines', name='Path 1'))
    # fig.add_trace(go.Scatter(x=path2[:0], y=path2[:,1], mode='lines', name='Path 2'))
    # fig.show()


def animate(positions: np.ndarray, confidences: np.ndarray, ppm: int=10):
    xs, ys = get_map_space(positions, ppm)
    dists = np.array([gaussian_2d(xs, ys, pos, std) for pos, std in zip(positions, confidences)])
    max_amp = np.max(dists, axis=None)
    # print(dists.shape)
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
    path = generate_path(100, 2)
    timestamps, positions, confidences = simulate(path)
    compare_paths(timestamps, path, positions)
    animate(positions, confidences)
