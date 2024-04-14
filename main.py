import numpy as np
from logging import getLogger

from utils import plot_bounded_path, generate_path, get_path_bounds
from kalman import simulate as kalman_simulate, plot_kalman_heatmaps
from ssp_sim import simulate as ssp_simulate, plot_ssp_heatmaps, decode_ssps


if __name__ == "__main__":
    np.random.seed(0)
    log = getLogger(__name__)
    T = 10
    dt = 0.001
    ssp_args = {'length_scale': 0.1}

    n_steps = int(T/dt)
    path = generate_path(n_steps, 2)
    bounds = get_path_bounds(path)

    ssp_space, ssps = ssp_simulate(path, **ssp_args)
    k_pos, cov = kalman_simulate(path, dt)
    plot_ssp_heatmaps(ssps, ssp_space, bounds, normalize=True)
    plot_kalman_heatmaps(k_pos, cov, bounds)

    log.info("Decoding...")
    pi_pos = decode_ssps(ssps, bounds, **ssp_args)
    timestamps = np.linspace(0, T, n_steps)
    plot_bounded_path(
        timestamps,
        [path, np.zeros((len(path), 2))],
        [k_pos, np.sqrt(np.diagonal(cov, axis1=1, axis2=2))],
        [pi_pos, np.zeros((len(pi_pos), 2))],
    )
