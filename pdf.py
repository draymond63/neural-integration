import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal



def gaussian2d(x: np.ndarray, y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    rv = multivariate_normal(mean, cov)
    domain = mesh(x, y)
    return rv.pdf(domain).reshape(len(x), len(y))


def get_cov(x: np.ndarray, y: np.ndarray, pdf: np.ndarray):
    """
    Calculate the covariance matrix of a domain given a pdf, of shape (domain_dim, domain_dim).

    Parameters
    ----------
    domain : np.ndarray
        The coordinates that correspond to the pdf, of shape (n_samples, domain_dim).
    pdf : np.ndarray
        The probability density function of the domain, of shape (n_samples,). Automatically normalized.

    Calculations
    ------------
    Expected value of x = E[x] = Σ[ x*p(x) ]
    Variance of x = E[x^2] - E[x]^2 = Σ[ x^2 * p(x) ] - (E[x])^2
    Covariance of x and y = E[xy] - E[x]E[y] = ΣΣ[ xy * p(x,y) ] - E[x]E[y]
    """
    npdf = pdf / np.sum(pdf)
    domain = mesh(x, y)
    dim = domain.shape[1]
    cov = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cov[i,j] = np.sum(domain[:,i] * domain[:,j] * npdf)
    mu = np.sum(domain * npdf[:, np.newaxis], axis=0)
    cov -= np.outer(mu, mu)
    return cov


def test_covariance(plot=False):
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    mean = np.array([1, 2])
    # cov = np.eye(2)
    cov = np.array([[3, 0], [0, 2]])
    pdf = gaussian2d(x, y, mean, cov).reshape(-1)
    if plot:
        go.Figure(data=go.Heatmap(x=x, y=y, z=pdf)).show()
    print(np.round(get_cov(x, y, pdf), 4))


def mesh(*ranges: np.ndarray) -> np.ndarray:
    return np.array(np.meshgrid(*ranges)).T.reshape(-1, len(ranges))


if __name__ == "__main__":
    test_covariance()
