import os
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

import pdf
from utils import plot_bounded_path, generate_path, get_path_bounds, get_bounded_space, plot_heatmaps
from kalman import simulate as kalman_simulate, plot_kalman_heatmaps
from ssp_sim import simulate as ssp_simulate, get_similarity_map, decode_ssps, get_ssp_stds


def compare_models(path: np.ndarray, dt: float, save=True, **ssp_args):
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    T = len(path) * dt
    bounds = get_path_bounds(path)
    encoder, ssps = ssp_simulate(path, **ssp_args)
    k_pos, k_cov = kalman_simulate(path, dt)

    xs, ys = get_bounded_space(bounds, ppm=50, padding=2)
    similarities = get_similarity_map(xs, ys, ssps, encoder)
    ssp_std = get_ssp_stds(similarities)
    ssp_pos = decode_ssps(ssps, bounds, **ssp_args)
    timestamps = np.linspace(0, T, len(path))
    plot_kalman_heatmaps(xs, ys, k_pos, k_cov)
    plot_heatmaps(xs, ys, similarities)    
    plot_bounded_path(
        timestamps,
        paths={
            'Truth': [path, np.zeros((len(path), 2))],
            'Kalman': [k_pos, pdf.get_stds(k_cov)],
            'SSP': [ssp_pos, ssp_std],
        }
    )
    if save:
        os.makedirs('data', exist_ok=True)
        np.savez(f"data/{run_stamp}.npz",
            path=path,
            dt=dt,
            xs=xs,
            ys=ys,
            k_pos=k_pos,
            k_cov=k_cov,
            ssps=ssps,
            ssp_pos=ssp_pos,
            similarities=similarities,
        )


def plot_from_file(filename: str):
    data = np.load(filename)
    path = data['path']
    T = len(path) * data['dt']
    timestamps = np.linspace(0, T, len(path))
    ssp_vars = get_ssp_stds(data['similarities'])
    bounds = get_path_bounds(path)
    xs, ys = get_bounded_space(bounds, ppm=50, padding=2)
    similarities = data['similarities']
    k_pos = data['k_pos']
    k_cov = data['k_cov']
    ssp_pos = data['ssp_pos']
    plot_kalman_heatmaps(xs, ys, k_pos, k_cov)
    plot_heatmaps(xs, ys, similarities)    
    plot_bounded_path(
        timestamps,
        paths={
            'Truth': [path, np.zeros((len(path), 2))],
            'Kalman': [k_pos, pdf.get_stds(k_cov)],
            'SSP': [ssp_pos, ssp_vars],
        }
    )


def compare_uncertainty(T: float, dt: float):
    num_steps = int(T / dt)
    path = generate_path(num_steps)
    tsteps = np.linspace(0, T, num_steps)
    bounds = get_path_bounds(path)

    encoder, ssps = ssp_simulate(path, length_scale=.1)
    k_pos, k_cov = kalman_simulate(path, dt)

    xs, ys = get_bounded_space(bounds, ppm=50, padding=2)
    similarities = get_similarity_map(xs, ys, ssps, encoder)
    ssp_vars = get_ssp_stds(similarities)
    k_stds = pdf.get_stds(k_cov)

    fig = go.Figure()
    fig.add_scatter(x=tsteps, y=k_stds[:, 0], name='Kalman X')
    fig.add_scatter(x=tsteps, y=k_stds[:, 1], name='Kalman Y')
    fig.add_scatter(x=tsteps, y=ssp_vars[:, 0], name='SSP X')
    fig.add_scatter(x=tsteps, y=ssp_vars[:, 1], name='SSP Y')
    fig.show()




if __name__ == "__main__":
    np.random.seed(0)
    T = 10
    dt = 0.01
    compare_uncertainty(T, dt)

    # num_steps = int(T / dt)
    # path = generate_path(num_steps)
    # compare_models(path, dt, length_scale=0.1)
