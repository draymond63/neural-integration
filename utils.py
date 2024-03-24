import numpy as np
import scipy 
from plotly.subplots import make_subplots
from typing import Tuple


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