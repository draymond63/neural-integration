import numpy as np

from encoders import HexagonalSSPSpace
from decoders import SSPSimilarityDecoder, train_decoder_net_sk
from utils import sample_domain_grid, generate_path, get_sample_spacing, plot_heatmaps, plot_path


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


def plot_ssp_heatmaps(ssps: np.ndarray, ssp_space: HexagonalSSPSpace, samples_per_dim=100, num_plots=9, **kwargs):
    domain_dim = ssp_space.domain_dim
    bounds = np.array([[-10, 10]] * domain_dim)
    t_spacing = get_sample_spacing(ssps, num_plots)
    plotted_ssps = ssps[::t_spacing]
    similarities = get_similarity_map(plotted_ssps, ssp_space, bounds, samples_per_dim=samples_per_dim)
    # TODO: Set boundaries properly
    xs = np.linspace(-10, 10, samples_per_dim)
    ys = np.linspace(-10, 10, samples_per_dim)
    plot_heatmaps(xs, ys, similarities, num_plots=9, **kwargs)


def get_similarity_map(ssps: np.ndarray, ssp_space: HexagonalSSPSpace, bounds, samples_per_dim=100):
    points = sample_domain_grid(bounds, samples_per_dim) # (samples_per_dim, domain_dim)
    ssp_grid = ssp_space.encode(points) # (timesteps, ssp_dim)
    pi_norms = np.linalg.norm(ssps, axis=1)[:, np.newaxis]
    ssp_output = ssps / np.where(pi_norms < 1e-6, 1, pi_norms) # (timesteps, ssp_dim)
    # Cosine similarity between encoded grid points and pi output (since both are unit vectors)
    similarities = ssp_grid @ ssp_output.T # (samples_per_dim**domain_dim, timesteps)
    similarities = similarities.reshape(samples_per_dim, samples_per_dim, -1).transpose(2,0,1) # (timesteps, samples_per_dim, samples_per_dim)
    return similarities



def decode(path: np.ndarray, ssps: np.ndarray, ssp_space: HexagonalSSPSpace):
    decoder = SSPSimilarityDecoder(path, ssp_space)
    return decoder.decode(ssps)


def grid_decoding(ssps: np.ndarray, ssp_space: HexagonalSSPSpace, bounds: np.ndarray, samples_per_dim=100):
    domain_dim = ssp_space.domain_dim
    similarities = get_similarity_map(ssps, ssp_space, bounds, samples_per_dim=samples_per_dim)
    coords = []
    for lb, ub in bounds.T:
        coords.append(np.linspace(lb, ub, samples_per_dim))
    coords = np.array(np.meshgrid(*coords)).T
    decoded = np.zeros((len(ssps), domain_dim))
    for i, grid in enumerate(similarities):
        ix, iy = np.unravel_index(np.argmax(grid), grid.shape)
        decoded[i] = coords[ix, iy, :]
    return decoded



if __name__ == "__main__":
    domain_dim = 2
    T = 20
    dt = 0.01
    num_steps = int(T/dt)
    length_scale = 0.1
    path = generate_path(num_steps, domain_dim)
    ssp_space, pi_out = simulate(path, length_scale=length_scale)
    plot_ssp_heatmaps(pi_out, ssp_space, normalize=True)

    # bounds = np.array([np.min(path, axis=0), np.max(path, axis=0)]).T
    # decoded = grid_decoding(pi_out, ssp_space, bounds)

    # decoder, hist = train_decoder_net_sk(ssp_space, bounds=bounds)
    # decoded = decoder.decode(pi_out)

    # decoded = decode(path, pi_out, ssp_space)
    # plot_path(np.linspace(0, T, num_steps), path, decoded)
