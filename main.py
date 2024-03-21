import nengo
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathintegration import PathIntegration
from sspspace import HexagonalSSPSpace
from utils import get_velocity_scale_factor


seed = 0
tau=0.05
dt = 0.001
T= 20
limit = 0.1
radius = 1
domain_dim = 2 # dim of space agent moves in 
timesteps = np.arange(0, T, dt)
n_timesteps = len(timesteps)

# Generate a random path
path = np.hstack([nengo.processes.WhiteSignal(T, high=limit, seed=seed+i).run(T,dt=dt) for i in range(domain_dim)])
shift_fun = lambda x, new_min, new_max: (new_max - new_min)*(x - np.min(x))/(np.max(x) - np.min(x))  + new_min
for i in range(path.shape[1]):
    path[:,i] = shift_fun(path[:,i], -0.9*radius,0.9*radius)    
pathlen = path.shape[0]
velocity_data = (1/dt)*( path[(np.minimum(np.floor(timesteps/dt) + 1, pathlen-1)).astype(int),:] -
                path[(np.minimum(np.floor(timesteps/dt), pathlen-2)).astype(int),:])


domain_dim = 2
initial_agent_position = path[0]


# Construct SSPSpace
ssp_space = HexagonalSSPSpace(domain_dim, 256)
d = ssp_space.ssp_dim

# function that returns (possible scaled by vel_scaling_factor) agent's velocity at time t
velocity_func, vel_scaling_factor = get_velocity_scale_factor(velocity_data, ssp_space, dt)


model = nengo.Network(seed=seed)

with model:
    # If running agent online instead, these will be output from other networks
    velocity = nengo.Node(velocity_func)
    init_state = nengo.Node(lambda t: ssp_space.encode(initial_agent_position) if t<0.05 else np.zeros(d))    
    pathintegrator = PathIntegration(ssp_space, 400, scaling_factor=vel_scaling_factor, stable=True)
    
    nengo.Connection(velocity, pathintegrator.velocity_input, synapse=0.01) 
    nengo.Connection(init_state, pathintegrator.input, synapse=None)
    
    pi_output_p = nengo.Probe(pathintegrator.output, synapse=0.05)


sim = nengo.Simulator(model)

with sim:
    sim.run(T)


pi_sim_path = ssp_space.decode(sim.data[pi_output_p], 'from-set','grid', 100)
np.savez('pathintegration.npz', path=path, pi_sim_path=pi_sim_path, T=T, dt=dt, seed=seed, limit=limit, radius=radius, domain_dim=domain_dim, initial_agent_position=initial_agent_position, velocity_data=velocity_data, vel_scaling_factor=vel_scaling_factor, ssp_space=ssp_space, d=d, timesteps=timesteps, n_timesteps=n_timesteps, pathlen=pathlen, pathintegrator_output=sim.data[pi_output_p])

fig = make_subplots(rows=2, cols=1, subplot_titles=("X", "Y"))
fig.add_trace(go.Scatter(x=sim.trange(), y=pi_sim_path[:,0], mode='lines', name='sim-x'), row=1, col=1)
fig.add_trace(go.Scatter(x=sim.trange(), y=path[:,0], mode='lines', name='real-x'), row=1, col=1)
fig.add_trace(go.Scatter(x=sim.trange(), y=pi_sim_path[:,1], mode='lines', name='sim-y'), row=2, col=1)
fig.add_trace(go.Scatter(x=sim.trange(), y=path[:,1], mode='lines', name='real-y'), row=2, col=1)
fig.show()
fig.to_image(format="png")


