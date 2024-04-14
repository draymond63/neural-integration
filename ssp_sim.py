import numpy as np

from encoders import HexagonalSSPSpace
from decoders import train_decoder_net_sk, train_decoder_net_tf, SSPDecoder
from utils import generate_path, get_sample_spacing, plot_heatmaps, plot_bounded_path, memory, get_path_bounds, get_bounded_space
import pdf


def simulate(path, noise_std=0.01, noise_pts=200, **kwargs):
    num_timesteps = len(path)
    deltas = np.diff(path, axis=0)
    encoder = HexagonalSSPSpace(domain_dim=2, **kwargs)

    if noise_std == 0:
        noise_pts = 1
    noise_dist = np.random.randn(noise_pts, 2) * noise_std

    x_t = encoder.encode(path[:1])
    ssps = [np.copy(x_t)]
    for i in range(num_timesteps - 1):
        delta = deltas[i]
        noisy_delta = delta + noise_dist
        dx_ssp = np.mean(encoder.encode(noisy_delta), axis=0, keepdims=True)
        x_t = x_t * dx_ssp
        ssps.append(x_t)
    ssps = np.array(ssps).squeeze()
    return encoder, ssps


def plot_ssp_heatmaps(xs, ys, similarities: np.ndarray, num_plots=9, **kwargs):
    t_spacing = get_sample_spacing(len(similarities), num_plots)
    plotted_sims = similarities[::t_spacing]
    plot_heatmaps(xs, ys, plotted_sims, num_plots=len(plotted_sims), **kwargs)


def get_similarity_map(xs, ys, ssps: np.ndarray, ssp_space: HexagonalSSPSpace, rescale=True):
    points = pdf.mesh(xs, ys)
    ssp_grid = ssp_space.encode(points) # (timesteps, ssp_dim)
    pi_norms = np.linalg.norm(ssps, axis=1)[:, np.newaxis]
    ssp_output = ssps / np.where(pi_norms < 1e-6, 1, pi_norms) # (timesteps, ssp_dim)
    # Cosine similarity between encoded grid points and pi output (since both are unit vectors)
    similarities = ssp_grid @ ssp_output.T # (len(xs)*len(ys))**domain_dim, timesteps)
    similarities = similarities.reshape(len(xs), len(ys), -1).transpose(2,0,1) # (timesteps, len(xs), len(ys))
    # Cosine similarity is in [-1, 1], so we rescale to [0, 1]
    if rescale:
        similarities += 1
        similarities /= 2
    return similarities


@memory.cache
def get_decoder(bounds: np.ndarray, tf=True, **encoder_kwargs) -> SSPDecoder:
    ssp_space = HexagonalSSPSpace(domain_dim=2, **encoder_kwargs)
    if tf:
        decoder, hist = train_decoder_net_tf(ssp_space, bounds=bounds)
    else:
        decoder, hist = train_decoder_net_sk(ssp_space, bounds=bounds)
    return decoder


def decode_ssps(ssps: np.ndarray, bounds: np.ndarray, **kwargs):
    decoder = get_decoder(bounds, **kwargs)
    decoded = decoder.decode(ssps)
    return decoded


if __name__ == "__main__":
    np.random.seed(0)
    T = 20
    dt = 0.01
    num_steps = int(T/dt)
    length_scale = 0.1
    path = generate_path(2000, 2)
    bounds = get_path_bounds(path)
    encoder, ssps = simulate(path, length_scale=length_scale)
    xs, ys = get_bounded_space(bounds, ppm=30, padding=2)
    similarities = get_similarity_map(xs, ys, ssps, encoder)
    plot_ssp_heatmaps(xs, ys, similarities, normalize=True)

    decoded_path = decode_ssps(ssps, bounds, length_scale=length_scale)
    timestamps = np.linspace(0, T, num_steps)
    stds = pdf.get_stds(pdf.get_covs(xs, ys, similarities.reshape(len(similarities), -1)))
    plot_bounded_path(
        timestamps,
        [path, np.zeros((len(path), 2))],
        [decoded_path, stds],
    )
