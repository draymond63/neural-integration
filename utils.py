import numpy as np
import scipy 
from plotly.subplots import make_subplots


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