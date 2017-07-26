import numpy as np
from numba import jit


@jit(nopython=True)
def kinship(X):
    """
    calculates the kinship matrix
    """
    n = float(X.shape[0])
    return np.dot(X, X.T) / n


def laplacian(A):
    """
    Calculates the graph Laplacian based on an adjacency matrix

    :param A: adjacency matrix  n x n
    :return: Laplacian   n x n
    """
    n, m = A.shape
    assert n == m, ValueError("Dimensions to not match")

    L = -A
    L[sp.diag_indices(n)] = 0

    s = sp.sum(L, axis=1)
    L[sp.diag_indices(n)] = -s

    return L


def normalized_laplacian(A):
    """
    calculates the normalized graph Laplacian

    :param A: Adjacency matrix   n x n
    :return: normalized Laplacian  n x n
    """
    n, m = A.shape
    assert n == m, ValueError("Dimensions to not match")

    A[np.diag_indices(n)] = 0.0

    d = np.sqrt(1.0 / np.sum(A, axis=1))
    d[np.isinf(d)] = 0.0

    L = laplacian(A)

    return (L * d).T * d


@jit(nopython=True)
def diffusion_graph_kernel(L, sigma=1.0):
    """
    calculates the graph diffusion kernel definde based on the normalized graph Laplacian
    defined in Smola and Kondor 2003

    K = exp(-sigma/2* L)

    :param L: normalized graph Laplacian
    :param kwargs: contains the hyperparameter (here named sigma)
    :return: the diffusion process kernel
    """

    # get hyperparameter
    return np.exp(-sigma * 0.5 * L)


@jit(nopython=True)
def p_random_walk_kernel(L, a=2.0, p=1.0):
    """
    calculates the p-step random walk graph kernel based on the normalized graph Laplacian
    defined by Smola and Kondor 2003

    :param L: normalized graph Laplacian
    :param kwargs: contains the hyperparameters (here named a and p; a >= 2)
    :return: the p-step random walk graph kernel
    """

    # get hyperparameters
    n = L.shape[0]
    assert a >= 2.0, ValueError("a must be greater or equal to 2")

    return np.power((a * np.identity(n) - L), p)


@jit(nopython=True)
def transform_kernel_to_distance(K):
    """
    transforms a kernel matrix to a distance matrix with
    P = I - K following Sokolov et al 2016

    :param K: Kernel
    :return: Distance matrix
    """

    return np.identity(K.shape[0]) - K