import nengo
import numpy as np
from plotly.subplots import make_subplots

from pathintegration import PathIntegration
from sspspace import HexagonalSSPSpace
from utils import get_velocity_scale_factor



def simulate(path: np.ndarray, dt=0.01, seed=0, ssp_dim=256, neurons=500, save=True, plot=True):
    duration = path.shape[0] * dt
    domain_dim = path.shape[1]
    ssp_space = HexagonalSSPSpace(domain_dim, ssp_dim)
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
    sim = nengo.Simulator(model)
    with sim:
        sim.run(duration)
    pi_sim_path = ssp_space.decode(sim.data[pi_output_p], 'from-set','grid', num_samples=100)

    if save:
        np.savez(
            'pathintegration.npz',
            path=path,
            pi_sim_path=pi_sim_path,
            dt=dt,
            seed=seed,
            domain_dim=domain_dim,
            velocity_data=velocity_data,
            vel_scaling_factor=vel_scaling_factor,
            ssp_space=ssp_space,
            pathintegrator_output=sim.data[pi_output_p]
        )
    if plot:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("X", "Y"))
        fig.add_scatter(x=sim.trange(), y=pi_sim_path[:,0], mode='lines', name='sim-x', marker=dict(color='blue'), row=1, col=1)
        fig.add_scatter(x=sim.trange(), y=path[:,0], mode='lines', name='real-x', marker=dict(color='red'), row=1, col=1)
        fig.add_scatter(x=sim.trange(), y=pi_sim_path[:,1], mode='lines', name='sim-y', showlegend=False, marker=dict(color='blue'), row=2, col=1)
        fig.add_scatter(x=sim.trange(), y=path[:,1], mode='lines', name='real-y', showlegend=False, marker=dict(color='red'), row=2, col=1)
        fig.show()
    return sim.data[pi_output_p]



def generate_path(T: float, dt: float, domain_dim: int, limit=0.1, radius=1, seed=0):
    path = nengo.processes.WhiteSignal(T, high=limit, seed=seed).run(T, domain_dim, dt)
    shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
    for i in range(path.shape[1]):
        path[:,i] = shift_fun(path[:,i], -0.9*radius,0.9*radius) 
    return path

if __name__ == "__main__":
    domain_dim = 2
    T = 30
    dt = 0.001
    path = generate_path(T, dt, domain_dim)
    simulate(path, dt)
