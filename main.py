import numpy as np
from logging import getLogger

from utils import plot_bounded_path, generate_path
from kalman import simulate as kalman_simulate, plot_kalman_heatmaps
from ssp_sim import simulate as ssp_simulate, plot_ssp_heatmaps, get_path_bounds


if __name__ == "__main__":
    log = getLogger(__name__)
    domain_dim = 2
    T = 10
    dt = 0.001
    n_steps = int(T/dt)
    path = generate_path(n_steps, domain_dim)
    bounds = get_path_bounds(path)

    ssp_space, pi_out = ssp_simulate(path)
    timestamps, k_pos, cov = kalman_simulate(path, dt)
    plot_ssp_heatmaps(pi_out, ssp_space, bounds, normalize=True)
    plot_kalman_heatmaps(k_pos, cov, bounds)
    # TODO: Replace decoding
    # log.info("Decoding...")
    # pi_pos = ssp_space.decode(pi_out, 'from-set', 'grid', num_samples=100)
    # plot_bounded_path(
    #     timestamps,
    #     [path, np.zeros((len(path), 2))],
    #     [k_pos, np.sqrt(np.diagonal(cov, axis1=1, axis2=2))],
    #     [pi_pos, np.zeros((len(pi_pos), 2))],
    # )
