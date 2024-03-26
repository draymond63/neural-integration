import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal



def gaussian2d(x: np.ndarray, y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    rv = multivariate_normal(mean, cov)
    domain = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    return rv.pdf(domain)


def get_cov(x: np.ndarray, y: np.ndarray, pdf: np.ndarray):
    """
    Calculate the covariance matrix of a domain given a pdf.

    Parameters
    ----------
    domain : np.ndarray
        The coordinates that correspond to the pdf. Of shape (n_samples, domain_dim).
    pdf : np.ndarray
        The probability density function of the domain. Of shape (n_samples,).
    """
    # Expected value of x = E[x] = Σ[ x*p(x) ]
    # Variance of x = E[x^2] - E[x]^2 = Σ[ x^2 * p(x) ] - (E[x])^2
    # Covariance of x and y = E[xy] - E[x]E[y] = ΣΣ[ xy * p(x,y) ] - E[x]E[y]
    # Covariance matrix = E[xx] - E[x]E[x]^T
    #                   = E[xx] - E[x]E[x]^T
    npdf = pdf / np.sum(pdf)
    domain = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    dim = domain.shape[1]
    cov = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cov[i,j] = expected_value(domain[:,i] * domain[:,j], npdf)
    # e_x = np.zeros(dim)
    # for i in range(dim):
    #     e_x[i] = expected_value_pdf(domain[:,i], npdf)
    # cov -= np.outer(e_x, e_x)
    # Equivalent to the following:
    cov -= np.outer(expected_value(domain, npdf[:, np.newaxis]), expected_value(domain, npdf[:, np.newaxis]))
    return cov


def expected_value(x: np.ndarray, pdf: np.ndarray, dx: float = 1.0):
    return np.sum(x * pdf) * dx


def test_covariance(plot=False):
    x = np.linspace(-10, 10, 20)
    y = np.linspace(-10, 10, 20)
    mean = np.ones(2)
    # cov = np.eye(2)
    cov = np.array([[1, 0.5], [0.5, 2]])
    pdf = gaussian2d(x, y, mean, cov).reshape(-1)
    if plot:
        go.Figure(data=go.Heatmap(x=x, y=y, z=pdf)).show()
    print(np.round(get_cov(x, y, pdf), 4))


if __name__ == "__main__":
    test_covariance()
