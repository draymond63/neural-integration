import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal



def gaussian2d(x: np.ndarray, y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    if np.all(cov == 0):
        cov = np.eye(2) * 1e-8
    rv = multivariate_normal(mean, cov)
    domain = mesh(x, y)
    return rv.pdf(domain).reshape(len(x), len(y))


def get_cov(x: np.ndarray, y: np.ndarray, pdf: np.ndarray):
    """
    Calculate the covariance matrix of shape (domain_dim, domain_dim) for a given a pdf,
    of shape (len(x)*len(y),). Currently, only supports 2D domains.

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
    assert pdf.shape == (len(x) * len(y),), f"pdf must be of shape ({len(x)*len(y)},), not {pdf.shape}"
    assert np.all(pdf >= 0), "pdf must be non-negative"
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

def get_covs(x: np.ndarray, y: np.ndarray, pdfs: np.ndarray) -> np.ndarray:
    assert len(pdfs.shape) == 2, "pdf must be of shape (n_samples, len(x)*len(y))"
    return np.array([get_cov(x, y, pdf) for pdf in pdfs])

def get_stds(covs: np.ndarray) -> np.ndarray:
    assert len(covs.shape) == 3, "covariances must be of shape (n_samples, 2, 2)"
    vars = np.copy(np.diagonal(covs, axis1=1, axis2=2))
    assert np.all(vars >= 0), "Variances must be non-negative"
    return np.sqrt(vars)


def test_covariance(plot=False):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    mean = np.array([-1, 2])
    # cov = np.eye(2)
    cov = np.array([[5, -5], [-5, 6]])
    pdf = gaussian2d(x, y, mean, cov).reshape(-1)
    if plot:
        go.Figure(data=go.Heatmap(x=x, y=y, z=pdf)).show()
    calc_cov = get_cov(x, y, pdf)
    assert np.allclose(calc_cov, cov, atol=0.1), f"Covariance calculation failed, got {calc_cov}"

    pdf = np.random.randn(len(x) * len(y)) * 0.1
    pdf -= np.min(pdf)
    cov = get_cov(x, y, pdf)
    assert np.all(np.diag(cov) > 0), "Variances matrix must be positive"


def mesh(*ranges: np.ndarray) -> np.ndarray:
    """
    Generate a meshgrid of points of shape `(len(range1)*len(range2)..., len(ranges))`.
    Typically, `len(ranges)` is the dimensionality of the domain

    e.g. mesh(x, y) -> [[x0,y0], [x0,y1], [x0,y2], ..., [x1,y0], [x1,y1], ...]
    """
    return np.array(np.meshgrid(*ranges)).T.reshape(-1, len(ranges))


if __name__ == "__main__":
    test_covariance()
