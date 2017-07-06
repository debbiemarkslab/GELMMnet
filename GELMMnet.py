"""
GELMMnet.py

Author:		Benjamin Schubert
Year:		2015
Group:		Debora Marks Group
Institutes:	Systems Biology, Harvard Medical School, 200 Longwood Avenue, Boston, 02115 MA, USA

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


The implementation is based on Barbara Rakitsch's implementation of LMM-Lasso (https://github.com/BorgwardtLab/LMM-Lasso)
and Artem Skolov's implementation of GELnet (https://github.com/cran/gelnet)

"""
from numba import jit
import numpy as np
import scipy as sp


def laplacian(A):
    """
    Calculates the graph Laplacian based on an adjacency matrix

    :param A: adjacency matrix  n x n
    :return: Laplacian   n x n
    """
    n, m = A.shape
    assert n == m, ValueError("Dimensions do not match")

    L = -A
    L[sp.diag_indices(n)] = 0

    s = sp.sum(L, axis=2)
    L[sp.diag_indices(n)] = -s

    return L


def normalized_laplacian(A):
    """
    calculates the normalized graph Laplacian

    :param A: Adjacency matrix   n x n
    :return: normalized Laplacian  n x n
    """
    n, m = A.shape
    assert n == m, ValueError("Dimensions do not match")

    A[sp.diag_indices(n)] = 0.0

    d = sp.sqrt(1.0 / sp.sum(A, axis=1))
    d[sp.isinf(d)] = 0.0

    L = laplacian(A)

    return (L*d).T*d


@jit(nopython=True)
def diffusion_graph_kernel(L, **kwargs):
    """
    calculates the graph diffusion kernel defined based on the normalized graph Laplacian
    defined in Smola and Kondor 2003

    K = exp(-sigma/2* L)

    :param L: normalized graph Laplacian
    :param kwargs: contains the hyperparameter (here named sigma)
    :return: the diffusion process kernel
    """

    # get hyperparameter
    sigma = kwargs.get("sigma", 1.0)

    return np.exp(-sigma*0.5*L)


@jit(nopython=True)
def p_random_walk_kernel(L, **kwargs):
    """
    calculates the p-step random walk graph kernel based on the normalized graph Laplacian
    defined by Smola and Kondor 2003

    :param L: normalized graph Laplacian
    :param kwargs: contains the hyperparameters (here named a and p; a >= 2)
    :return: the p-step random walk graph kernel
    """

    # get hyperparameters
    n = L.shape[0]
    a = kwargs.get("a", 2.0)
    p = kwargs.get("p", 1.0)
    assert a >= 2.0, ValueError("a must be greater or equal to 2")

    return np.power((a*np.identity(n) - L), p)


@jit(nopython=True)
def transform_kernel_to_distance(K):
    """
    transforms a kernel matrix to a distance matrix with
    P = I - K following Sokolov et al 2016

    :param K: Kernel
    :return: Distance matrix
    """

    return np.identity(K.shape[0]) - K


@jit(nopython=True)
def _eval_neg_log_likelihood(ldelta, Uy, S):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.

    :param ldelta: log-transformed ratio sigma_gg/sigma_ee
    :param Uy: transformed outcome: n_s x 1
    :param S: eigenvectors of K: n_s
    :return: negative log likelihood value
    """
    n_s = Uy.shape[0]
    delta = np.exp(ldelta)

    # evaluate log determinant
    Sd = S + delta
    ldet = np.sum(np.log(Sd))

    # evaluate the variance
    Sdi = 1.0 / Sd
    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    # evaluate the negative log likelihood
    nLL = 0.5 * (n_s * np.log(2.0 * sp.pi) + ldet + n_s + n_s * np.log(ss))

    return nLL


@jit(nopython=True)
def _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m):
    """
    calculates the gelnet objective

     1/(2*n)*Sum(y_i - S)^2 + l1*sum(|w|) +l2/2*w^T*P*w

    :return: gelnet regularized loss function
    """
    loss = np.sum(np.power(y - S, 2.0))
    reg_l1 = np.sum(np.abs(w))
    reg_l2 = np.dot(w.T, Pw)

    return loss/(2.0*n) + l1*reg_l1 + 0.5*l2*reg_l2


@jit(nopython=True)
def _snap_threshold(residual, gamma):
    """
    soft-thresholding function to accelerate lasso regression

    :param residual: residual to should be snapped back to zero
    :param gamma: threshold boundary
    :return: snapped residual
    """
    if np.less(np.fabs(residual), gamma):
        return 0.0
    elif np.less(residual, 0.0):
        return residual + gamma
    else:
        return residual + gamma


@jit(nopython=True)
def _update_wj(X, y, P, w, l1, l2, S, Pw, n, m, j):
    """
    Update rule based on coordinate descent
    """
    numerator = np.sum(X[:, j] * (y - S + X[:, j] * w[j]))

    numerator /= n
    numerator -= l2 * (Pw[j] - P[j, j] * w[j])

    # snap value to zero again
    numerator = _snap_threshold(numerator, l1)

    if np.equal(numerator, 0.0):
        return 0.0

    denom = np.sum(np.power(X[:, j], 2))
    denom /= n
    denom += l2*P[j, j]

    return numerator / denom


@jit(nopython=True)
def _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, debug):

    obj_old = _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m)

    # start optimization
    # optimize for max_iter steps
    for i in range(max_iter):
        # update each weight individually
        for j in range(m):
            w_old = w[j]
            w[j] = _update_wj(X, y, P, l1, l2, S, Pw, n, m, j)
            wj_dif = w[j] - w_old

            # update fit
            if np.not_equal(wj_dif, 0):
                S += X[:, j] * wj_dif
                Pw += P[:, j] * wj_dif

        # update bias
        old_b = b
        b = np.sum(y - (y - b)) / n
        b_diff = b - old_b

        # update fits accordingly
        S += b_diff

        # calculate objective and test for convergence
        obj = _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m)
        abs_dif = np.fabs(obj - obj_old)

        # optimization converged?
        if np.less(abs_dif / np.fabs(obj_old), eps):
            break
        else:
            obj_old = obj

    return w, b


class GELMMnet(object):
    """
    Generalized network-based elastic-net linear mixed model


    1) We first infer sigma_g and sigma_e based on a Null-model following Kang et al 2010.
    2) We than rotate y and S based on the eigendecomposition of K following Rakitsch et al 2013
       and Schelldorfer et al 2011.
    3) We than fit the weights with coordinate descent as the transformation renders the problem a "simple"
       elastic-net inference problem
    """

    def __init__(self, y, X, K):

        # first check for correct input
        assert X.shape[0] == y.shape[0], ValueError('dimensions do not match')
        assert K.shape[0] == K.shape[1], ValueError('dimensions do not match')
        assert K.shape[0] == X.shape[0], ValueError('dimensions do not match')

        self.__n, self.__m =X.shape

        if y.ndim == 1:
            y = sp.reshape(y, (self.__n, 1))

        self.__y = y
        self.__X = X
        self.__K = K
        self.__SUX = None
        self.__SUy = None
        self.__w = np.zeros(self.__m)
        self.__b = np.mean(y)
        self.__ldelta = 0

    def train_null_model(self, numintervals=100, ldeltamin=-5, ldeltamax=5):
        """
        Optimizes sigma_g and simga_e based on a grid search using the approach
        proposed by Khang et al. 2010

        """

        n = self.__n
        S, U = sp.linalg.eigh(self.__K)
        Uy = sp.dot(U.T, self.__y)

        nllgrid = sp.ones(numintervals + 1) * np.inf
        ldeltagrid = sp.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        nllmin = sp.inf

        # initial grid search
        for i in np.arange(numintervals + 1):
            nllgrid[i] = _eval_neg_log_likelihood(ldeltagrid[i], Uy, S)

        # find minimum
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        # more accurate search around the minimum of the grid search
        for i in sp.arange(numintervals - 1) + 1:
            if nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]:
                ldeltaopt, nllopt, iter, funcalls = sp.optimize.brent(_eval_neg_log_likelihood,
                                                                      (Uy, S),
                                                                      (ldeltagrid[i - 1],
                                                                       ldeltagrid[i],
                                                                       ldeltagrid[i + 1]),
                                                                      full_output=True)
                if nllopt < nllmin:
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt

        # train lasso on residuals
        self.__ldelta = ldeltaopt_glob
        delta0 = sp.exp(ldeltaopt_glob)
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = sp.sqrt(Sdi)
        SUX = sp.dot(U.T, self.__X)
        self.__SUX = SUX * sp.tile(Sdi_sqrt, (self.__m, 1)).T
        SUy = sp.dot(U.T, self.__y)
        self.__SUy = SUy * sp.reshape(Sdi_sqrt, (n, 1))

    def train(self, P, l1=1.0, l2=1.0, eps=1e-5, max_iter=10000, debug=False):
        """
        Train on the rotated data
        """

        if self.__SUX is None:
            self.train_null_model()

        X = self.__SUX
        y = self.__SUy
        w = self.__w
        b = self.__b
        n, m = self.__SUX.shape

        assert P.shape[0] == n, ValueError("P's dimensions do not match")

        # initial parameter estimates
        S = sp.dot(X, w)
        Pw = sp.dot(P, w)

        w, b = _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w=w, b=b, debug=debug)

        self.__w = w
        self.__b = b

    def predict(self, X_tilde):
        """
        predicts the phenotype based on the trained model

        following Rasmussen and Williams 2006

        :param X_tilde: test matrix n_test x m
        :return: y_tilde the predicted phenotype
        """
        if self.__SUX is None:
            RuntimeError("The model has not been trained yet")

        assert X_tilde.shape[1] == self.__m, ValueError("Dimension of input data does not match")

        w = self.__w
        b = self.__b
        n = self.__n
        y = self.__y
        X = self.__X
        # calculate kniship matrix for the new data point
        n_test = X_tilde.shape[0]
        delta = sp.exp(self.__ldelta)
        idx = w.nonzero()[0]

        # calculate kinship matrices
        K = self.__K
        K_vt = 1.0/n * sp.dot(X_tilde, self.__X.T)

        if idx.shape[0] == 0:
            return sp.dot(K_vt, sp.linalg.solve(K + delta*sp.eye(n), y))

        y_v = sp.dot(X_tilde[:, idx], w[idx]) + b + sp.dot(K_vt, sp.linalg.solve(K + delta * sp.eye(n),
                                                                             y - sp.dot(X[:, idx], w[idx]) + b))
        return y_v

    def post_selection_analysis(self, alternative='twosided', alpha=0.05, UMAU=False, compute_intervals=False):
        """
        implements the post selection analysis proposed by

        Lee, J. D., Sun, D. L., Sun, Y., & Taylor, J. E. (2016).
        Exact post-selection inference, with application to the lasso.
        The Annals of Statistics, 44(3), 907-927.


        See: https://github.com/selective-inference/Python-software

        :param str alternative: The type of test statistic to calculate ['twosoded', 'onesided']
        :param float alpha: The (1-alpha)*100% selective confidence intervals
        :param bool UMAU: Whethter UMAU intervals should be calculated
        :return: DataFrame with one entry per active variable. Columns are
                 'variable', 'pval', 'lasso', 'onestep', 'lower_trunc', 'upper_trunc', 'sd'.

        """




