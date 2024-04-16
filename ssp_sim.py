import os
import numpy as np
import tensorflow as tf
from scipy import stats

from encoders import HexagonalSSPSpace
from decoders import train_decoder_net_tf, SSPDecoder
from utils import generate_path, plot_heatmaps, plot_bounded_path, get_path_bounds, get_bounded_space, apply_kernel
import pdf


def simulate(path, noise=0.003, noise_pts=1000, **kwargs):
    num_timesteps = len(path)
    deltas = np.diff(path, axis=0)
    encoder = HexagonalSSPSpace(domain_dim=2, **kwargs)

    if noise == 0:
        noise_pts = 1
    noise_dist = np.random.randn(noise_pts, 2) * noise

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

def get_ssp_stds(similarities: np.ndarray):
    std = np.sqrt(get_ssp_variance(similarities))
    return np.stack([std, std], axis=1)

def get_ssp_variance(similarities: np.ndarray, method='edge'):
    """Calculates the estimated variance of the """
    assert method in ['edge', 'var', 'entropy']
    if method == 'edge':
        return get_edge_variance(similarities)
    elif method == 'var':
        variance = np.var(similarities, axis=(1, 2))
        return 430*(variance - variance[0])
    elif method == 'entropy':
        entropy = stats.entropy(similarities, axis=(1, 2))
        return 240*(entropy[0] - entropy)

def get_edge_variance(similarities: np.ndarray):
    kernel = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
    edges = apply_kernel(similarities, kernel)
    avg_edge = np.abs(edges).mean(axis=(1, 2))
    return _edge_to_var_transform(avg_edge)

def _edge_to_var_transform(arr, A=0.11058, B=7.271e-08):
    """Edge strength is logarithmic, but variance is linear. This map is a rough fit to observation."""
    return A*(np.exp((arr[0] - arr)/B) - 1)


def get_decoder(bounds: np.ndarray, save_as: str=None, load=True, **encoder_kwargs) -> SSPDecoder:
    ssp_space = HexagonalSSPSpace(domain_dim=2, **encoder_kwargs)
    if load and save_as is not None and os.path.exists(save_as):
        model = tf.keras.models.load_model(save_as)
        decoder = SSPDecoder(bounds, model, encoder=ssp_space)
        return decoder
    decoder, hist = train_decoder_net_tf(ssp_space, bounds=bounds)
    if save_as is not None:
        decoder.decoder_network.save(save_as)
    return decoder


def decode_ssps(ssps: np.ndarray, bounds: np.ndarray, **kwargs):
    decoder = get_decoder(bounds, save_as='decoder.keras', **kwargs)
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
    plot_heatmaps(xs, ys, similarities)

    decoded_path = decode_ssps(ssps, bounds, length_scale=length_scale)
    timestamps = np.linspace(0, T, num_steps)
    stds = get_ssp_var(similarities)

    plot_bounded_path(
        timestamps,
        paths={
            'Truth': [path, np.zeros((len(path), 2))],
            'SSP': [decoded_path, stds],
        }
    )
