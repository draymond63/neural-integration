import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.stats import spearmanr
from tqdm import tqdm
from typing import Dict

import pdf
from utils import plot_bounded_path, generate_path, get_path_bounds, get_bounded_space, plot_heatmaps
from kalman import simulate as kalman_simulate, plot_kalman_heatmaps
from ssp_sim import simulate as ssp_simulate, get_similarity_map, decode_ssps, get_ssp_stds, get_uncertainty_metric


def compare_models(path: np.ndarray, dt: float, noise=1e-3, save=True, **ssp_args):
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    T = len(path) * dt
    bounds = get_path_bounds(path)
    encoder, ssps = ssp_simulate(path, noise=noise, **ssp_args)
    k_pos, k_cov = kalman_simulate(path, dt, measurement_noise=noise)

    xs, ys = get_bounded_space(bounds, ppm=50, padding=2)
    similarities = get_similarity_map(xs, ys, ssps, encoder)
    ssp_stds = get_ssp_stds(similarities)
    k_stds = pdf.get_stds(k_cov)
    ssp_pos = decode_ssps(ssps, bounds, **ssp_args)
    timestamps = np.linspace(0, T, len(path))
    plot_kalman_heatmaps(xs, ys, k_pos, k_cov)
    plot_heatmaps(xs, ys, similarities)    
    plot_bounded_path(
        timestamps,
        paths={
            'Truth': [path, np.zeros((len(path), 2))],
            'Kalman': [k_pos, k_stds],
            'SSP': [ssp_pos, ssp_stds],
        }
    )
    data = dict(
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
    if save:
        os.makedirs('data', exist_ok=True)
        np.savez(f"data/{run_stamp}.npz", **data)
    return data


def plot_model_comparison(data: Dict[str, np.ndarray]):
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


def compare_uncertainty_metrics(path: np.ndarray, dt: float, noise=1e-3, methods=('sharpness', 'var', 'entropy'), plot_metrics=False, plot_maps=False) -> pd.DataFrame:
    encoder, ssps = ssp_simulate(path, noise=noise, length_scale=.1)
    k_pos, k_cov = kalman_simulate(path, dt, process_noise=noise)

    bounds = get_path_bounds(path)
    xs, ys = get_bounded_space(bounds, ppm=50, padding=5)
    similarities = get_similarity_map(xs, ys, ssps, encoder)
    k_var = k_cov[:, 0, 0]
    var_esimates = {method: get_uncertainty_metric(similarities, method=method) for method in methods}

    if plot_maps:
        plot_kalman_heatmaps(xs, ys, k_pos, k_cov)
        plot_heatmaps(xs, ys, similarities)
    if plot_metrics:
        T = len(path) * dt
        tsteps = np.linspace(0, T, len(path))
        fig = go.Figure()
        fig.add_scatter(x=tsteps, y=k_var, name='Kalman')
        for method, estimate in var_esimates.items():
            srcc = spearmanr(k_var, estimate)
            fig.add_scatter(x=tsteps, y=estimate, name=f'SSP-{method.capitalize()} ({srcc.correlation:.4f})')
            print(f'{method}: {srcc.correlation:.8f} ({srcc.pvalue:.3E})')
        fig.show()
    df = pd.DataFrame(var_esimates)
    df['kalman'] = k_var
    df['noise'] = noise
    return df


def compare_metrics_at_noise_levels(path: np.ndarray, dt: float, noise_range: np.ndarray, save=True):
    dfs = []
    for noise in tqdm(noise_range):
        df = compare_uncertainty_metrics(path, dt, noise)
        dfs.append(df)
    df = pd.concat(dfs)
    if save:
        timestamp = datetime.now().strftime("%y%m%d-%H%M")
        df.to_csv(f'data/{noise_range[0]}-{noise_range[-1]}-{len(noise_range)}_{timestamp}.csv')
    return df


def plot_metric_comparison(df: pd.DataFrame, every=1):
    truth = df['kalman']
    methods = ('sharpness', 'var', 'entropy')
    noises = sorted(df['noise'].unique())[::every]
    noise_titles = [f"1E{np.log10(n):.2f}" for n in noises]
    fig = make_subplots(rows=len(methods), cols=len(noises), shared_xaxes=True, row_titles=methods, column_titles=noise_titles)
    for r, method in enumerate(methods, 1):
        for c, n in enumerate(noises, 1):
            mask = df['noise'] == n
            values = df[method][mask]
            values -= values.min()
            values /= values.max()
            srcc = spearmanr(truth[mask], values)
            fig.add_scatter(y=values, row=r, col=c)
            fig.add_annotation(text=f'{srcc.correlation:.3f}', xref='x domain', yref='y domain', x=0.5, y=-0.1, showarrow=False, row=r, col=c)
        srcc = spearmanr(truth, df[method])
        print(f'{method} (all): {srcc.correlation:.8f} ({srcc.pvalue:.3E})')
    fig.update_layout(showlegend=False, title='Uncertainty Evolution at Different Noise Levels')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()

def plot_correlations(df: pd.DataFrame):
    truth = df['kalman']
    methods = ('sharpness', 'var', 'entropy')
    correlations = {}
    noises = sorted(df['noise'].unique())
    for method in methods:
        correlations[method] = []
        for noise in noises:
            mask = df['noise'] == noise
            values = df[method][mask]
            srcc = spearmanr(truth[mask], values)
            correlations[method].append(abs(srcc.correlation))
    fig = go.Figure()
    for method, values in correlations.items():
        fig.add_scatter(x=noises, y=values, name=method, mode='lines+markers')
    fig.update_xaxes(type='log', title='Noise Standard Deviation')
    fig.update_yaxes(title='Spearman Correlation Magnitude')
    fig.show()


if __name__ == "__main__":
    T = 20
    dt = 0.01
    num_steps = int(T / dt)
    path = generate_path(num_steps)
    # df = compare_uncertainty_metrics(path, dt, 1e-1, plot_metrics=True, plot_maps=True)
    noise_range = np.logspace(-5, -1, 20)
    df = compare_metrics_at_noise_levels(path, dt, noise_range)

    # df = pd.read_csv('data/uncertainty_metrics.csv')
    plot_metric_comparison(df, every=2)
    plot_correlations(df)

    # num_steps = int(T / dt)
    # path = generate_path(num_steps)
    # compare_models(path, dt, length_scale=0.1)
