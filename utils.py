import numpy as np
import scipy 
from scipy.stats import qmc
from joblib import Memory
from plotly.subplots import make_subplots
from typing import Tuple


memoize = Memory(location='cache', verbose=0)


def sparsity_to_x_intercept(d, p):
    sign = 1
    if p > 0.5:
        p = 1.0 - p
        sign = -1
    return sign * np.sqrt(1-scipy.special.betaincinv((d-1)/2.0, 0.5, 2*p))


def get_velocity_scale_factor(velocity_data: np.ndarray, ssp_space, dt: float):
    pathlen = velocity_data.shape[0]
    real_freqs = (ssp_space.phase_matrix @ velocity_data.T)
    vel_scaling_factor = 1/np.max(np.abs(real_freqs))
    vels_scaled = velocity_data*vel_scaling_factor
    velocity_func = lambda t: vels_scaled[int(np.minimum(np.floor(t/dt), pathlen-2))]
    return velocity_func, vel_scaling_factor


def plot_heatmaps(x, y, zs, num_plots=9, normalize=False):
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    fig = make_subplots(rows=rows, cols=cols)
    for i in range(num_plots):
        fig.add_heatmap(
            x=x,
            y=y,
            z=zs[i],
            zmin=0,
            row=i//cols+1,
            col=i%cols+1,
            zmax=1 if normalize else None,
            showscale=normalize,
        )
    fig.update_layout(showlegend=False)
    fig.show()


def get_sample_spacing(arr: np.ndarray, num_samples: int):
    return np.ceil(len(arr) / num_samples).astype(int)


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


def plot_path(ts: float, *paths: np.ndarray):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    dims = ['x', 'y']
    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for idx, path in enumerate(paths):
        design_params = dict(mode='lines', showlegend=len(paths) > 1, col=1)
        design_params['marker'] = dict(color=colors[idx])
        for dim, dim_label in enumerate(dims):
            design_params['row'] = dim + 1
            fig.add_scatter(x=ts, y=path[:,dim], name=f"Path {idx}", **design_params)
            design_params['showlegend'] = False

    for i, dim_label in enumerate(dims, 1):
        fig.update_yaxes(title_text=f"{dim_label.capitalize()}", row=i, col=1)
    fig.update_xaxes(title_text="Time (s)", row=len(dims), col=1)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.show()


def plot_bounded_path(ts: np.ndarray, *paths: Tuple[np.ndarray, np.ndarray]):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    dims = ['x', 'y']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    bound_params = dict(line=dict(dash='dash'), showlegend=False, mode='lines', col=1)

    for idx, (path, stdevs) in enumerate(paths):
        design_params = dict(mode='lines', showlegend=len(paths) > 1, col=1)
        design_params['marker'] = dict(color=colors[idx])
        bound_params['marker'] = dict(color=colors[idx])
        for dim, dim_label in enumerate(dims):
            design_params['row'] = dim + 1
            bound_params['row'] = dim + 1
            fig.add_scatter(x=ts, y=path[:,dim], name=f"Path {idx}", **design_params)
            fig.add_scatter(x=ts, y=path[:,dim] + stdevs[:,dim], **bound_params)
            fig.add_scatter(x=ts, y=path[:,dim] - stdevs[:,dim], **bound_params)
            design_params['showlegend'] = False

    for i, dim_label in enumerate(dims, 1):
        fig.update_yaxes(title_text=f"{dim_label.capitalize()}", row=i, col=1)
    fig.update_xaxes(title_text="Time (s)", row=len(dims), col=1)
    fig.show()



def sample_domain(bounds, num_samples):
    sampler = qmc.Sobol(d=bounds.shape[0]) 
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    u_sample_points = sampler.random(num_samples)
    sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
    return sample_points


def sample_domain_grid(bounds, samples_per_dim=100):
    domain_dim = bounds.shape[0]
    xxs = np.meshgrid(*[np.linspace(bounds[i,0], bounds[i,1], samples_per_dim) for i in range(domain_dim)])
    retval = np.array([x.reshape(-1) for x in xxs]).T
    assert retval.shape[1] == domain_dim, f'Expected {domain_dim}d data, got {retval.shape[1]}d data'
    return retval


def make_good_unitary(dim, eps=1e-3, rng=np.random, mul=1):
    a = rng.rand((dim - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * mul * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


def conjugate_symmetry(K):
    d = K.shape[0]
    F = np.zeros((d*2 + 1,K.shape[1]))#, dtype="complex")
    F[0:d,:] = K
    F[-d:,:] = np.flip(-F[0:d,:],axis=0)
    return np.fft.ifftshift(F, axes=0)


def vecs_from_phases(K):
    d = K.shape[0]
    F = np.ones((d*2+2, K.shape[1]), dtype="complex")
    F[1:d+1,:] = np.exp(1.j*K)
    F[-d:,:] = np.flip(np.conj(F[1:d+1,:]),axis=0)
    F[0,:] = 1
    F[d+1,:] = 1
    return F
