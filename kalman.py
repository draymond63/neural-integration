import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Agent:
    def __init__(self, init_state: np.ndarray, dt: float, process_noise=1e-3, measurement_noise=1e-2):
        if len(init_state) != 4:
            raise ValueError("Initial state must be a 4D vector [x, y, vx, vy]")
        self.x = np.array(init_state)
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.P = np.eye(4) # Initial covariance matrix
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
    return positions, confidence


def get_map_space(positions: np.ndarray, ppm=1):
    """Returns 2x2 array of maps"""
    max_dims = np.max(positions, axis=0) + 1
    min_dims = np.min(positions, axis=0)
    points = max_dims - min_dims
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


def plot_heatmap(positions: np.ndarray, confidences: np.ndarray, ppm=10):
    xs, ys = get_map_space(positions, ppm)
    fig = make_subplots(rows=len(positions), cols=1)
    for i, (pos, std) in enumerate(zip(positions, confidences), 1):
        dist = gaussian_2d(xs, ys, pos, std)
        fig.add_trace(go.Heatmap(z=dist, type='heatmap', colorscale='Viridis'), row=i, col=1)
    fig.update_layout(
        title="Agent Position Confidence Over Time",
        xaxis_title="X Position",
        yaxis_title="Y Position"
    )
    fig.show()


if __name__ == "__main__":
    path = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 2]])
    positions, confidences = simulate(path)
    plot_heatmap(positions, confidences)
