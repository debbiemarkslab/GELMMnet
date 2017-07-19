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
import functools

import numpy as np
import scipy as sp
import pandas as pd
import multiprocessing as mp

from numba import jit

from scipy.stats import norm
from collections import OrderedDict

from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error

import regreg.api as rr



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

    A[sp.diag_indices(n)] = 0.0

    d = sp.sqrt(1.0 / sp.sum(A, axis=1))
    d[sp.isinf(d)] = 0.0

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
    claculates the gelnet objective

     1/(2*n)*Sum(y_i - S)^2 + l1*sum(|w|) +l2/2*w^T*P*w

    :return: gelnet regularized loss function
    """
    loss = np.sum(np.power(y[:, 0] - S, 2.0))
    reg_l1 = np.sum(np.abs(w))
    reg_l2 = np.dot(w.T, Pw)

    return loss / (2.0 * n) + l1 * reg_l1 + 0.5 * l2 * reg_l2


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
        return residual - gamma


@jit(nopython=True)
def _update_wj(X, y, P, w, l1, l2, S, Pw, n, m, j):
    """
    Update rule based on coordinate descent

    :param X:
    :param y:
    :param P:
    :param l1:
    :param l2:
    :param S:
    :param Pw:
    :param n:
    :param m:
    :return:
    """
    numerator = np.sum(X[:, j] * (y[:, 0] - S + X[:, j] * w[j]))

    numerator /= n
    numerator -= l2 * (Pw[j] - P[j, j] * w[j])

    # snap value to zero again
    numerator = _snap_threshold(numerator, l1)

    if np.equal(numerator, 0.0):
        return 0.0
    denom = np.sum(np.power(X[:, j], 2))
    denom /= n
    denom += l2 * P[j, j]

    return numerator / denom


@jit(nopython=True)
def _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, debug):
    obj_old = _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m)
    old_b = b
    # start optimization
    # optimize for max_iter steps
    for i in range(max_iter):
        # update each weight individually
        for j in range(m):
            w_old = w[j]
            # _update_wj(X, y, P, w, l1, l2, S, Pw, n, m, j)
            w[j] = _update_wj(X, y, P, w, l1, l2, S, Pw, n, m, j)
            wj_dif = w[j] - w_old

            # update fit
            if np.not_equal(wj_dif, 0):
                S += X[:, j] * wj_dif  # dont know if that is correct
                Pw += P[:, j] * wj_dif

        # update bias
        # TODO: hier stimmt was nicht
        old_b = b
        b = np.mean(y[:, 0] - (S - b))
        b_diff = b - old_b

        # update fits accordingly
        S += b_diff

        # calculate objective and test for convergence
        obj = _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m)
        abs_dif = np.fabs(obj - obj_old)

        # optimization converged?
        if np.less(abs_dif, eps):
            break
        else:
            obj_old = obj

    return w, b


def _calc_pval(y, A, b, v, sigma):
    """
    calculates p-value based on defined polyhedral

    :param y:
    :param A:
    :param b:
    :param v:
    :param sigma:
    :return:
    """
    z = np.sum(v*y)
    vv = np.sum(np.power(v, 2))
    sd = sigma * np.sqrt(vv)

    rho = np.dot(A, v) / vv
    vec = (b - np.dot(A, y) + rho * z) / rho

    vlo = np.max(vec[rho > 0])
    vup = np.min(vec[rho < 0])


    # calc p-value directly
    return _truncatednorm_surv(z, 0, vlo, vup, sd), vlo, vup


def _truncatednorm_surv(z, mean, a, b, sd):
    def ff(z):
        return ((np.power(z, 2) + 5.575192695 * z + 12.7743632) /
                (np.power(z, 3) * np.sqrt(2 * np.pi) + 14.38718147 * z * z + 31.53531977 * z + 2 * 12.77436324))

    zz = max(min(z, b), a)
    if mean == np.inf:
        return 1
    if mean == -np.inf:
        return 0

    zz = (zz - mean) / sd
    aa = (a - mean) / sd
    bb = (b - mean) / sd
    p = (norm.cdf(bb) - norm.cdf(zz)) / (norm.cdf(bb) - norm.cdf(aa))
    # test if we generated any NaNs or infs, if so we approximate these values
    if np.isnan(p) or np.isinf(p):
        # Returns Prob(Z>z | Z in [a,b]), where mean can be a vector, based on
        # A UNIFORM APPROXIMATION TO THE RIGHT NORMAL TAIL INTEGRAL, W Bryc
        # Applied Mathematics and Computation
        # Volume 127, Issues 23, 15 April 2002, Pages 365--374
        # https://math.uc.edu/~brycw/preprint/z-tail/z-tail.pdf
        term1 = np.exp(zz * zz)
        if aa > -np.inf:
            term1 = ff(aa) * np.exp(-(np.power(aa, 2) - np.power(zz, 2)) / 2)
        term2 = 0
        if bb < np.inf:
            term2 = ff(bb) * np.exp(-(np.power(bb, 2) - np.power(zz, 2)) / 2)
        p = (ff(zz) - term2) / (term1 - term2)
        if np.isnan(p):
            return np.NaN
        p = min(1.0, max(0.0, p))
    return p


def _calc_interval(y, A, b, v, sigma, alpha, gridrange=[-100, 100], gridpts=100, griddepth=1, flip=False):
    z = np.sum(v*y)
    vv = np.sum(np.power(v, 2))
    sd = sigma * np.sqrt(vv)

    rho = np.dot(A, v) / vv
    vec = (b - np.dot(A, y) + rho * z) / rho
    vlo = np.max(vec[rho > 0])
    vup = np.min(vec[rho < 0])

    xg = np.linspace(gridrange[0]*sd, gridrange[1]*sd, num=gridpts)

    fun = functools.partial(lambda z, vlo, vup, sd, x: _truncatednorm_surv(z, x, vlo, vup, sd), z, vlo, vup, sd)
    int = _grid_search(xg, fun, alpha/2., 1 - alpha/2., gridpts, griddepth)
    tailarea = [fun(int[0]), 1 - fun(int[1])]

    if flip:
        return [-x for x in int[::-1]], tailarea[::-1]
    return int, tailarea


def _grid_search(grid, fun, val1, val2, gridpts=100, griddepth=1):
    vals = np.array([fun(x) for x in grid])

    ii = np.where(vals >= val1)[0]
    jj = np.where(vals <= val2)[0]

    if not ii.size:
        return [grid[-1], np.inf]
    if not jj.size:
        return [-np.inf, grid[0]]

    i1 = int(np.min(ii))
    i2 = int(np.max(jj))

    if i1 == 0:
        lo = -np.inf
    else:
        # might be wrong index due to R starting at 1 (perhaps all -1)
        lo = _grid_bsearch(grid[i1-1], grid[i1], fun, val1, gridpts, griddepth - 1, below=True)

    if i2 == (len(grid)-1):
        hi = np.inf
    else:
        hi = _grid_bsearch(grid[i2], grid[i2+1], fun, val2, gridpts, griddepth -1, below=False)
    #print("CI", lo, hi)
    return [lo, hi]


def _grid_bsearch(l, r, fun, val, gridpts=100, griddepth=0, below=True):
    left = l
    right = r
    #print("left",l,"right",r,"val",val)
    n = gridpts
    depth = 0
    while depth <= griddepth:
        grid = np.linspace(left, right, num=n)
        vals = np.array(list(map(fun, grid)))

        if below:
            ii = np.where(vals >= val)[0]
            #print(below, " ii ", ii)
            if not ii.size:
                #print(below, " ii empty", grid[-1])
                return grid[-1]

            i0 = int(np.min(ii))
            if not i0:
                #print(below, " i0 min", grid[0])
                return grid[0]
            left = grid[i0 - 1]
            right = grid[i0]

        else:
            ii = np.where(vals <= val)[0]
            #print(below, " ii ", ii)
            if not ii.size:
                #print(below, " ii empty", grid[0])
                return grid[0]

            i0 = int(np.max(ii))
            if i0 >= (n - 1):
                #print(below, " i0 max", grid[-1])
                return grid[-1]

            left = grid[i0]
            right = grid[i0+1]
        depth += 1
    #print(below, " bound ", left if below else right)
    return left if below else right


@jit(nopython=True)
def max_l1(y, X):
    """
    returns the upper limit of l1 (i.e., smallest value that yields a model with all zero weights)

    """
    n, m = X.shape
    assert n == len(y), ValueError("Dimensions do not match.")

    b = np.mean(y)
    xy = np.mean(np.dot(X.T, (y - b)), axis=1)
    return np.max(np.fabs(xy))


def _param_search(alpha, lmabda, metric, ytrain, Xtrain, ytest, Xtest, K, P, eps, max_iter):
    """
    evaluates one parameter option
    :param alpha:
    :param lmabda:
    :param metric:
    :param ytrain:
    :param Xtrain:
    :param ytest:
    :param Xtest:
    :return: metric, l1, and l2 penalty parameters
    """
    l1 = lmabda * alpha
    l2 = alpha * (1. - lmabda)
    lmm=False
    if K is not None:
        lmm=True
        K = np.identity(Xtrain.shape[0])

    glmmnet = GELMMnet(ytrain, Xtrain, K)
    glmmnet.train(P, lmm=lmm, l1=l1, l2=l2, eps=eps, max_iter=max_iter)
    ypred = glmmnet.predict(Xtest)
    return metric(ytest, ypred), l1, l2


def kfold(X, y, P, K=None, centered=False, nfold=10, alpha_nof=100,
          ratio_nof=100, eps=1e-5, max_iter=10000, cpu=1, metric=mean_squared_error, get_best=max):
        """
        kfold fit and variance estimate for post-selection analysis

        we are fitting:

        l(Xw,y) + alpha*ratio*L1(w) + alpha*(1-ratio)*0.5*L2(w,P)

        :param P:
        :param nfold:
        :param lmm:
        :param l1_nof:
        :param ratio_nof:
        :param eps:
        :param max_iter:
        :return: trained model with estimated sd
        """
        def generate_grid():
            for train_id, test_id in cv.split(X):
                Xtrain, Xtest = X[train_id], X[test_id]
                ytrain, ytest = y[train_id], y[test_id]
                Ktrain = K[train_id][:, train_id]
                for a in alphas:
                    for r in ratios:
                        yield a, r, metric, ytrain, Xtrain, ytest, Xtest, Ktrain, P, eps, max_iter

        cv = KFold(n_splits=nfold)
        pool = mp.Pool(processes=cpu)

        ratios = np.linspace(0., 1., num=ratio_nof)

        #centering data to avoid offset
        if not centered:
            y = scale(y, with_std=False)
            X = scale(X, with_std=False)

        alpha_ceil = max_l1(y, X)
        alphas = np.linspace(0., alpha_ceil, num=alpha_nof)

        grid_result = pool.starmap(_param_search, generate_grid())

        #find arg max
        _,l1,l2 = get_best(grid_result,key=lambda x: x[0])
        g = GELMMnet(y, X, K)
        lmm = False
        if K is not None:
            lmm = True
            g.train_null_model()

        _, w = g.train(P, lmm=lmm, l1=l1, l2=l2, max_iter=max_iter, eps=eps)
        ypred = g.predict(X)
        df = np.sum(w != 0) - 1
        sigma = np.sqrt(np.sum(np.power(y - ypred, 2)))/(len(y) - df)
        g.set_sigma = sigma
        return g


class GELMMnet(object):
    """
    Generalized network-based elastic-net linear mixed model

    \min_\beta~\frac{1}{2}\sum_{i=1}^N (y_i - X_i\beta)^2 + \lambda_1\|\beta\|_1 + \frac{\lambda_2}{2}\beta^TP\beta

    1) We first infer sigma_g and sigma_e based on a Null-model following Kang et al 2010.
    2) We than rotate y and S based on the eigendecomposition of K following Rakitsch et al 2013
       and Schelldorfer et al 2011.
    3) We than fit the weights with coordinate descent as the transformation renders the problem a "simple"
       elastic-net inference problem

    """

    def __init__(self, y, X, K):

        # first check for correct input
        assert X.shape[0] == y.shape[0], ValueError('dimensions do not match X({}), y({})'.format(X.shape[0]),y.shape[0])
        assert K.shape[0] == K.shape[1], ValueError('dimensions do not match')
        assert K.shape[0] == X.shape[0], ValueError('dimensions do not match')

        v = np.isnan(y)
        keep = True ^ v
        if v.sum():
            print("Cleaning the phenotype vector by removing %d individuals...\n" % (v.sum()))
            y = y[keep]
            X = X[keep, :]
            K = K[keep, :][:, keep]

        self.__keep = keep
        self.__n, self.__m = X.shape

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
        self.__l1 = 1.0
        self.__l2 = 1.0
        self.__P = None
        self.__sigma = np.mean(y)
        self.__summary = None

    @property
    def summary(self):
        return self.__summary

    @property
    def w(self):
        return self.__w

    @property
    def b(self):
        return self.__b

    @property
    def ldelta(self):
        return self.__ldelta

    @property
    def sigma(self):
        return self.__sigma

    def set_sigma(self, sigma):
        self.__sigma = sigma

    def train_null_model(self, numintervals=100, ldeltamin=-20, ldeltamax=20, debug=False):
        """
        Optimizes sigma_g and simga_e based on a grid search using the approach
        proposed bei Khang et al. 2010

        """
        n = self.__n
        S, U = sp.linalg.eigh(self.__K)

        if debug:
            print("Optimizing")
            print("U:", U)
            print("U.T", U.T)
            print("y", self.__y)

        Uy = sp.dot(U.T, self.__y)
        nllgrid = sp.ones(numintervals + 1) * np.inf
        ldeltagrid = sp.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        nllmin = sp.inf

        # initial grid search
        for i in np.arange(numintervals + 1):
            nllgrid[i] = _eval_neg_log_likelihood(ldeltagrid[i], Uy, S)
            if debug:
                print("Init grid:", nllgrid[i], ldeltagrid[i])

        # find minimum
        nll_min = nllgrid.min()
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
                    if debug:
                        print("Second grid:", nllopt, ldeltaopt)
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

        if debug:
            print("delta0", delta0)
            print("Sdi_sqrt", Sdi_sqrt)
            print("SUX", SUX)
            print("SUy", SUy)

    def train_old(self, P, l1=1.0, l2=1.0, eps=1e-5, max_iter=1000, debug=False):
        """
            Train complete model
        """

        if self.__SUX is None:
            self.train_null_model()

        X = self.__SUX
        y = self.__SUy
        w = self.__w
        b = self.__b
        n, m = self.__SUX.shape
        assert P.shape[0] == m, ValueError("P's dimensions do not match")

        # initial parameter estimates
        S = sp.dot(X, w) + b
        Pw = sp.dot(P, w)

        w, b = _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w=w, b=b, debug=debug)
        if debug:
            print("Optimal values:", w, b)
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
        y = self.__y
        X = self.__X

        # calculate kniship matrix for the new data point
        n_test = X_tilde.shape[0]
        n_train = X.shape[0]
        delta = sp.exp(self.__ldelta)
        idx = w.nonzero()[0]
        Xtmp = np.concatenate((X, X_tilde), axis=0)

        # calculate Covariance matrices
        K = 1. / Xtmp.shape[0] * sp.dot(Xtmp, Xtmp.T)
        idx_tt = sp.arange(n_train)
        idx_vt = sp.arange(n_train, n_test + n_train)

        K_tt = K[idx_tt][:, idx_tt]
        K_vt = K[idx_vt][:, idx_tt]

        if idx.shape[0] == 0:
            return sp.dot(K_vt, sp.linalg.solve(K_tt + delta * sp.eye(n_train), y))

        y_v = sp.dot(X_tilde[:, idx], w[idx]) + b + sp.dot(K_vt, sp.linalg.solve(K_tt + delta * sp.eye(n_train),
                                                                                 y[:, 0] - sp.dot(X[:, idx],
                                                                                                  w[idx]) + b))
        return y_v

    def train(self, P, lmm=True, l1=1.0, l2=1.0, eps=1e-5, max_iter=10000):
        """
        Trains the model with regreg

        :param P:
        :param l1:
        :param l2:
        :param eps:
        :param max_iter:
        :return:
        """
        self.__l1 = l1
        self.__l2 = l2
        self.__P = P

        if self.__SUX is None and lmm:
            self.train_null_model()

        X = self.__SUX if lmm else self.__X
        y = self.__SUy[:,0] if lmm else self.__y[:,0]
        n, m = X.shape

        assert P.shape[0] == m, ValueError("P's dimensions do not match")

        loss = rr.squared_error(X, y)

        grouping = rr.quadratic_loss(m, Q=P, coef=l2)

        sparsity = rr.l1norm(m, lagrange=l1)

        problem = rr.container(loss, grouping, sparsity)
        solver = rr.FISTA(problem)
        obj_vals = solver.fit(max_its=max_iter, tol=eps)
        self.__w = solver.composite.coefs
        return obj_vals, self.__w

    def post_selection_analysis(self, lmm=True, alpha=0.1, gridrange=[-100,100], compute_intervals=False):
        """
        implements the post selection analysis proposed by

        Lee, J. D., Sun, D. L., Sun, Y., & Taylor, J. E. (2016).
        Exact post-selection inference, with application to the lasso.
        The Annals of Statistics, 44(3), 907-927.


        See: https://github.com/selective-inference/Python-software

        :param float alpha: The (1-alpha)*100% selective confidence intervals
        :param bool UMAU: Whethter UMAU intervals should be calculated
        :return: DataFrame with one entry per active variable. Columns are
                 'variable', 'pval', 'lasso', 'onestep', 'lower_trunc', 'upper_trunc', 'sd'.

        """
        if self.__SUX is None:
            RuntimeError("The model has not been trained yet")

        X = self.__SUX if lmm else self.__X
        y = self.__SUy[:,0] if lmm else self.__y[:,0]
        w = self.__w
        P = self.__P
        l2 = self.__l2
        l1 = self.__l1
        sigma = self.__sigma if self.__sigma is not None else np.std(y)

        # dont know if this is correct

        active = np.nonzero(w != 0)[0]
        active_signs = np.sign(w[active])

        # compute the hessian of the active set M
        # H(M) = (XT_M*X_M + l2*P)^-1
        XM = X[:, active]

        H = (np.dot(np.transpose(X), X) + l2*P)
        H_AA = H[active][:, active]
        H_AAinv = np.linalg.pinv(H_AA)

        D = np.diag(active_signs)

        '''
        The active set is defined as:

        A1(M,s) = -diag(s)(X^T_M*X_M + l2*P)^-1*X^T_M
        b1(M,s) = -l1*diag(s)(X^T_M*X_M + l2*P)^-1*s
        '''
        A = np.dot(D, np.dot(H_AAinv, np.transpose(XM)))

        b = l1 * np.dot(np.dot(D, H_AAinv), active_signs)

        ################################################################################################################
        # P-value and CI calculations

        k = len(active)
        M = np.dot(H_AAinv, np.transpose(XM))

        result = []
        for j in range(k):

            vj = M[j]
            mj = np.sqrt(np.sum(np.power(vj, 2)))
            vj = vj / mj
            sign = np.sign(np.sum(vj*y))
            vj = sign * vj

            #calculate p-value
            _pval, vlo, vup = _calc_pval(y, A, b, vj, sigma)
            vmat = vj * mj * sign

            if compute_intervals:
                _interval, tailarea = _calc_interval(y, A, b, vj, sigma, alpha,
                                                     gridrange=gridrange, flip=sign == -1)

                ci = [x*mj for x in _interval]
            else:
                ci = [np.nan, np.nan]

            sd = sigma*np.sqrt(np.sum(np.power(vmat, 2)))
            coef0 = np.dot(vmat, y)
            result.append((active[j],
                           _pval,
                           coef0,
                           w[active[j]],
                           coef0 / sd,
                           ci[0],
                           ci[1],
                           tailarea[0],
                           tailarea[1],
                           sd))

        df = pd.DataFrame(index=active,
                      data=OrderedDict([(n, d) for n, d in zip(['variable',
                                                         'pval',
                                                         'coef0',
                                                         'beta',
                                                         'Zscore',
                                                         'lower_confidence',
                                                         'upper_confidence',
                                                         'lower_trunc',
                                                         'upper_trunc',
                                                         'sd'],
                                                        np.array(result).T)])).set_index('variable')
        self.__summary = df
        return df



