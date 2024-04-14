import numpy as np
from logging import getLogger

import pdf
from utils import plot_bounded_path, generate_path, get_path_bounds, get_bounded_space
from kalman import simulate as kalman_simulate, plot_kalman_heatmaps
from ssp_sim import simulate as ssp_simulate, get_similarity_map, plot_ssp_heatmaps, decode_ssps


if __name__ == "__main__":
    np.random.seed(0)
    log = getLogger(__name__)
    T = 10
    dt = 0.001
    ssp_args = {'length_scale': 0.1}

    n_steps = int(T/dt)
    path = generate_path(n_steps, 2)
    bounds = get_path_bounds(path)

    encoder, ssps = ssp_simulate(path, **ssp_args)
    k_pos, k_cov = kalman_simulate(path, dt)

    xs, ys = get_bounded_space(bounds, ppm=30)
    similarities = get_similarity_map(xs, ys, ssps, encoder)
    plot_ssp_heatmaps(xs, ys, similarities, normalize=True)
    plot_kalman_heatmaps(xs, ys, k_pos, k_cov)

    log.info("Decoding...")
    ssp_pos = decode_ssps(ssps, bounds, **ssp_args)
    ssp_covs = pdf.get_covs(xs, ys, similarities.reshape(len(similarities), -1))
    timestamps = np.linspace(0, T, n_steps)
    plot_bounded_path(
        timestamps,
        [path, np.zeros((len(path), 2))],
        [k_pos, pdf.get_stds(k_cov)],
        [ssp_pos, pdf.get_stds(ssp_covs)],
    )
