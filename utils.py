import numpy as np
from scipy.stats import qmc
from scipy.signal import convolve2d
from plotly.subplots import make_subplots
from typing import Tuple, Dict



def plot_heatmaps(x, y, zs, num_plots=9, offset=True):
    cols = int(np.ceil(np.sqrt(num_plots)))
    rows = int(np.ceil(num_plots / cols))
    if len(zs) > num_plots:
        spacing = get_sample_spacing(len(zs), num_plots)
        zs = zs[::spacing]
    if offset:
        zs = zs - np.min(zs)
    fig = make_subplots(rows=rows, cols=cols)
    for i in range(num_plots):
        fig.add_heatmap(
            x=x,
            y=y,
            z=zs[i],
            zmin=0,
            row=i//cols+1,
            col=i%cols+1,
            zmax=None,
            showscale=False,
        )
    fig.update_layout(showlegend=False)
    fig.show()


def get_sample_spacing(og_len: int, num_samples: int):
    return np.ceil(og_len / num_samples).astype(int)


def generate_path(n_steps, domain_dim=2, smoothing_window=200):
    """Generate a random continuous path in 2D space, of shape (n_steps, domain_dim)"""
    path = np.cumsum(np.random.randn(n_steps, domain_dim), axis=0)
    smoothing_window = min(smoothing_window, n_steps)
    window = np.hanning(smoothing_window)
    path[:, 0] = np.convolve(path[:, 0], window, mode='same')
    path[:, 1] = np.convolve(path[:, 1], window, mode='same')
    path = 2 * path / np.max(np.abs(path), axis=0)
    path -= path[0]
    return path


def get_path_bounds(path: np.ndarray):
    return np.array([np.min(path, axis=0), np.max(path, axis=0)]).T


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


def plot_bounded_path(ts: np.ndarray, paths: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    dims = ['x', 'y']
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    bound_params = dict(line=dict(dash='dash'), showlegend=False, mode='lines', col=1)

    for idx, (name, (path, stdevs)) in enumerate(paths.items()):
        design_params = dict(mode='lines', showlegend=len(paths) > 1, col=1)
        design_params['marker'] = dict(color=colors[idx])
        bound_params['marker'] = dict(color=colors[idx])
        for dim, dim_label in enumerate(dims):
            design_params['row'] = dim + 1
            bound_params['row'] = dim + 1
            fig.add_scatter(x=ts, y=path[:,dim], name=name, **design_params)
            fig.add_scatter(x=ts, y=path[:,dim] + stdevs[:,dim], **bound_params)
            fig.add_scatter(x=ts, y=path[:,dim] - stdevs[:,dim], **bound_params)
            design_params['showlegend'] = False

    for i, dim_label in enumerate(dims, 1):
        fig.update_yaxes(title_text=f"{dim_label.capitalize()}", row=i, col=1)
    fig.update_xaxes(title_text="Time (s)", row=len(dims), col=1)
    fig.show()



def sample_domain_rng(bounds, num_samples):
    sampler = qmc.Sobol(d=bounds.shape[0]) 
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    u_sample_points = sampler.random(num_samples)
    sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
    return sample_points


def get_path_bounds(path: np.ndarray, decimals=1):
    bounds = np.array([np.min(path, axis=0), np.max(path, axis=0)]).T
    bounds = np.round(bounds, decimals=decimals)
    return bounds

def get_bounded_space(bounds: np.ndarray, ppm=1, padding=0.1):
    """Returns 2x2 array of maps"""
    assert bounds.shape[0] == 2
    delta = bounds[:,1] - bounds[:,0]
    delta += delta * padding
    points = (delta * ppm).astype(int)
    return [np.linspace(lb - padding, ub + padding, p) for (lb, ub), p in zip(bounds, points)]


def apply_kernel(sequence, kernel):
    res = [None] * sequence.shape[0]
    for i, f in enumerate(sequence):
        npdf = f / np.sum(f)
        res[i] = convolve2d(npdf, kernel, mode='valid')
    res = np.array(res)
    return res


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
