import nengo
import numpy as np
from plotly.subplots import make_subplots

from pathintegration import PathIntegration
from sspspace import HexagonalSSPSpace
from utils import memoize, get_velocity_scale_factor, plot_heatmaps, get_sample_spacing


@memoize.cache()
def simulate(path: np.ndarray, dt=0.01, seed=0, neurons=500, plot=True, **kwargs):
    domain_dim = path.shape[1]
    ssp_space = HexagonalSSPSpace(domain_dim, **kwargs)
    d = ssp_space.ssp_dim # Might differ from ssp_dim unfortunately
    velocity_data = np.diff(path, axis=0) / dt
    # function that returns (possible scaled by vel_scaling_factor) agent's velocity at time t
    velocity_func, vel_scaling_factor = get_velocity_scale_factor(velocity_data, ssp_space, dt)
    model = nengo.Network(seed=seed)
    with model:
        velocity = nengo.Node(velocity_func)
        init_state = nengo.Node(lambda t: ssp_space.encode(path[int(np.floor(t/dt))]) if t<0.05 else np.zeros(d))    
        pathintegrator = PathIntegration(ssp_space, neurons, scaling_factor=vel_scaling_factor, stable=True)
        nengo.Connection(velocity, pathintegrator.velocity_input, synapse=0.01) 
        nengo.Connection(init_state, pathintegrator.input, synapse=None)
        pi_output_p = nengo.Probe(pathintegrator.output, synapse=0.05)
    # Simulate!
    duration = path.shape[0] * dt
    sim = nengo.Simulator(model)
    with sim:
        sim.run(duration)
    pi_sim_path = ssp_space.decode(sim.data[pi_output_p], 'from-set','grid', num_samples=100)

    if plot:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("X", "Y"))
        fig.add_scatter(x=sim.trange(), y=pi_sim_path[:,0], mode='lines', name='sim-x', marker=dict(color='blue'), row=1, col=1)
        fig.add_scatter(x=sim.trange(), y=path[:,0], mode='lines', name='real-x', marker=dict(color='red'), row=1, col=1)
        fig.add_scatter(x=sim.trange(), y=pi_sim_path[:,1], mode='lines', name='sim-y', showlegend=False, marker=dict(color='blue'), row=2, col=1)
        fig.add_scatter(x=sim.trange(), y=path[:,1], mode='lines', name='real-y', showlegend=False, marker=dict(color='red'), row=2, col=1)
        fig.show()
    return ssp_space, sim.data[pi_output_p]


def plot_pi_heatmaps(pi_output: np.ndarray, ssp_space: HexagonalSSPSpace, samples_per_dim=100, num_plots=9, **kwargs):
    points = ssp_space.get_sample_points(samples_per_dim=samples_per_dim, method='grid') # (timesteps, domain_dim)
    ssp_grid = ssp_space.encode(points) # (timesteps, ssp_dim)
    pi_norms = np.linalg.norm(pi_output, axis=1)[:, np.newaxis]
    ssp_output = pi_output / np.where(pi_norms < 1e-6, 1, pi_norms) # (timesteps, ssp_dim)
    # Cosine similarity between encoded grid points and pi output (since both are unit vectors)
    t_spacing = get_sample_spacing(ssp_output, num_plots)
    similarities = ssp_grid @ ssp_output[::t_spacing].T # (samples_per_dim*samples_per_dim, timesteps)
    similarities = similarities.reshape(samples_per_dim, samples_per_dim, -1).transpose(2,0,1) # (timesteps, samples_per_dim, samples_per_dim)
    # domain_bounds = np.min(points, axis=0), np.max(points, axis=0)
    # TODO: Set boundaries properly
    xs = np.linspace(-10, 10, samples_per_dim)
    ys = np.linspace(-10, 10, samples_per_dim)
    plot_heatmaps(xs, ys, similarities, num_plots=9, **kwargs)



def generate_path(T: float, dt: float, domain_dim: int, limit=0.1, radius=1, seed=0):
    path = nengo.processes.WhiteSignal(T, high=limit, seed=seed).run(T, domain_dim, dt)
    shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
    for i in range(path.shape[1]):
        path[:,i] = shift_fun(path[:,i], -0.9*radius,0.9*radius) 
    return path

if __name__ == "__main__":
    domain_dim = 2
    T = 20
    dt = 0.001
    length_scale = .5
    path = generate_path(T, dt, domain_dim)
    ssp_space, pi_out = simulate(path, dt, length_scale=length_scale)
    plot_pi_heatmaps(pi_out, ssp_space, normalize=True)

    # trange = np.arange(0, T, dt)
    # pi_sim_path = ssp_space.decode(pi_out, 'from-set','grid', num_samples=100)
    # fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("X", "Y"))
    # fig.add_scatter(x=trange, y=pi_sim_path[:,0], mode='lines', name='sim-x', marker=dict(color='blue'), row=1, col=1)
    # fig.add_scatter(x=trange, y=path[:,0], mode='lines', name='real-x', marker=dict(color='red'), row=1, col=1)
    # fig.add_scatter(x=trange, y=pi_sim_path[:,1], mode='lines', name='sim-y', showlegend=False, marker=dict(color='blue'), row=2, col=1)
    # fig.add_scatter(x=trange, y=path[:,1], mode='lines', name='real-y', showlegend=False, marker=dict(color='red'), row=2, col=1)
    # fig.show()
