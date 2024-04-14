import numpy as np

from encoders import HexagonalSSPSpace
from utils import generate_path, get_sample_spacing, plot_heatmaps, plot_path, get_path_bounds, get_bounded_space
from pdf import mesh


np.random.seed(0)


def simulate(path, length_scale=0.1, **kwargs):
    num_timesteps = len(path)
    deltas = np.diff(path, axis=0)
    encoder = HexagonalSSPSpace(domain_dim=2, length_scale=length_scale, **kwargs)

    x_t = encoder.encode(path[:1])
    ssps = [np.copy(x_t)]
    for i in range(num_timesteps - 1):
        dx, dy = deltas[i]
        dx_ssp = encoder.encode([[dx,dy]])
        x_t = x_t * dx_ssp
        ssps.append(x_t)
    ssps = np.array(ssps).squeeze()
    return encoder, ssps


def plot_ssp_heatmaps(ssps: np.ndarray, ssp_space: HexagonalSSPSpace, bounds: np.ndarray, ppm=30, num_plots=9, **kwargs):
    t_spacing = get_sample_spacing(ssps, num_plots)
    plotted_ssps = ssps[::t_spacing]
    similarities = get_similarity_map(plotted_ssps, ssp_space, bounds, ppm=ppm)
    # TODO: Set boundaries properly
    xs, ys = get_bounded_space(bounds, ppm)
    plot_heatmaps(xs, ys, similarities, num_plots=9, **kwargs)


def get_similarity_map(ssps: np.ndarray, ssp_space: HexagonalSSPSpace, bounds: np.ndarray, ppm=5):
    xs, ys = get_bounded_space(bounds, ppm)
    points = mesh(xs, ys)
    ssp_grid = ssp_space.encode(points) # (timesteps, ssp_dim)
    pi_norms = np.linalg.norm(ssps, axis=1)[:, np.newaxis]
    ssp_output = ssps / np.where(pi_norms < 1e-6, 1, pi_norms) # (timesteps, ssp_dim)
    # Cosine similarity between encoded grid points and pi output (since both are unit vectors)
    similarities = ssp_grid @ ssp_output.T # (samples_per_dim**domain_dim, timesteps)
    similarities = similarities.reshape(len(xs), len(ys), -1).transpose(2,0,1) # (timesteps, samples_per_dim, samples_per_dim)
    return similarities



if __name__ == "__main__":
    domain_dim = 2
    T = 20
    dt = 0.01
    num_steps = int(T/dt)
    length_scale = 0.1
    path = generate_path(num_steps, domain_dim)
    bounds = get_path_bounds(path)
    ssp_space, pi_out = simulate(path, length_scale=length_scale)
    plot_ssp_heatmaps(pi_out, ssp_space, bounds, normalize=True)

    # bounds = np.array([np.min(path, axis=0), np.max(path, axis=0)]).T
    # decoded = grid_decoding(pi_out, ssp_space, bounds)

    # decoder, hist = train_decoder_net_sk(ssp_space, bounds=bounds)
    # decoded = decoder.decode(pi_out)

    # decoded = decode(path, pi_out, ssp_space)
    # plot_path(np.linspace(0, T, num_steps), path, decoded)
