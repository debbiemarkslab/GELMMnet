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
, Artem Skolov's implementation of GELnet (https://github.com/cran/gelnet), and Ryan Tibshirani et al.'s implementation
of selectiveInference (https://cran.r-project.org/web/packages/selectiveInference/index.html)

"""
from pathos.pools import ParallelPool as Pool

from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy as sp

from sklearn.model_selection import KFold
from sklearn.preprocessing import scale

from GELMMnet.utility.inference import max_l1, _eval_neg_log_likelihood, _optimize_gelnet, _predict, _parameter_search
from GELMMnet.utility.postselection import _calc_interval, _calc_pval


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

    def __init__(self, y, X, K=None, intercept=True, standardize=True):

        # first check for correct input
        assert X.shape[0] == y.shape[0], ValueError('dimensions do not match X({}), y({})'.format(X.shape[0]),y.shape[0])

        v = np.isnan(y)
        keep = True ^ v
        if v.sum():
            print("Cleaning the phenotype vector by removing %d individuals...\n" % (v.sum()))
            y = y[keep]
            X = X[keep, :]
            if K is not None:
                K = K[keep, :][:, keep]
                assert K.shape[0] == K.shape[1], ValueError('dimensions do not match')
                assert K.shape[0] == X.shape[0], ValueError('dimensions do not match')

        # instead of fitting a intercept we will  standardize the data
        if standardize or intercept:
            y = scale(y, with_std=False)
            X = scale(X, with_std=False)

        self.__keep = keep
        self.__n, self.__m = X.shape

        if y.ndim == 1:
            y = sp.reshape(y, (self.__n, 1))

        self.__isStandardized = standardize
        self.__islmm = K is not None
        self.__isIntercept = intercept or standardize

        self.__y = y
        self.__X = X
        self.__K = K
        self.__P = None

        self.__SUX = None
        self.__SUy = None

        self.__w = np.zeros(self.__m)
        self.__b = np.mean(y) if not self.__isIntercept else 0
        self.__sigma = np.mean(y)
        self.__ldelta = 0

        self.__l1 = 1.0
        self.__l2 = 1.0

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

    @property
    def Xtilde(self):
        return self.__SUX

    @property
    def ytilde(self):
        return self.__SUy

    def fit_null_model(self, numintervals=100, ldeltamin=-20, ldeltamax=20, debug=False):
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

    def fit(self, P, l1=1.0, l2=1.0, eps=1e-5, max_iter=1000):
        """
            Train complete model
        """

        self.__l2 = l2
        self.__P = P
        self.__l1 = l1

        if self.__SUX is None and self.__islmm:
            self.fit_null_model()

        X = self.__SUX if self.__islmm else self.__X
        y = self.__SUy if self.__islmm else self.__y
        n, m = X.shape
        w = self.__w
        b = self.__b

        assert P.shape[0] == m, ValueError("P's dimensions do not match")

        # initial parameter estimates
        S = sp.dot(X, w) + b
        Pw = sp.dot(P, w)

        w, b = _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, self.__isIntercept)

        self.__w = w
        self.__b = b
        return b, w

    def kfoldFit(self, P, nfold=10, alpha_nof=100, ratio_nof=100, eps=1e-5, max_iter=10000, cpu=1, debug=False):
        """
        optimizes l1 and l2 based on k-fold cv with grid search minimizing the MSE

        """

        def generate_grid():
            for i, (train_id, test_id) in enumerate(cv.split(X)):
                Xtrain, Xtest = X[train_id], X[test_id]
                ytrain, ytest = y[train_id], y[test_id]

                for a in alphas:
                    for r in ratios:
                        yield i, a, r, w, b, delta, self.__isIntercept, ytrain, Xtrain, ytest, Xtest, P, eps, max_iter

        self.__P = P
        if self.__SUX is None and self.__islmm:
            self.fit_null_model()

        pool = Pool(nodes=cpu)
        cv = KFold(n_splits=nfold)

        X = self.__SUX if self.__islmm else self.__X
        y = self.__SUy if self.__islmm else self.__y
        n, m = X.shape
        w = self.__w
        b = self.__b
        delta = np.exp(self.__ldelta)

        alpha_ceil = max_l1(y, X)
        ratios = np.linspace(0.0, 1., num=ratio_nof) # that should actually be skewed towards 1.
        alphas = np.linspace(0.0, alpha_ceil, num=alpha_nof, endpoint=False)

        grid_result = pool.map(_parameter_search, generate_grid())

        # summarize grid search results
        sum_res = {}
        for fold, error, l1, l2 in grid_result:
            sum_res.setdefault((l1, l2), []).append(error)

        # find best l1, l2 pair across the folds
        (l1, l2), error = min(sum_res.items(), key=lambda x: np.mean(x[1]))
        if debug:
            print("Best Parameters:", l1, l2, "With error:", np.mean(error))

        # fitting on whole date with best parameter pair
        S = sp.dot(X, w) + b
        Pw = sp.dot(P, w)

        w, b = _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, self.__isIntercept)

        self.__w = w
        self.__b = b

        # estimate error
        ypred = self.predict(X)
        df = np.sum(w != 0) - 1
        sigma = np.sqrt(np.sum(np.power(y - ypred, 2)))/(len(y) - df)
        self.__sigma = sigma

        return l1, l2, sigma

    def predict(self, X_tilde):
        """
        predicts the phenotype based on the trained model

        following Rasmussen and Williams 2006

        :param X_tilde: test matrix n_test x m
        :return: y_tilde the predicted phenotype
        """

        assert X_tilde.shape[1] == self.__m, ValueError("Dimension of input data does not match")

        w = self.__w
        b = self.__b
        y = self.__y
        X = self.__X
        delta = np.exp(self.__ldelta)

        # calculate kniship matrix for the new data point
        return _predict(X, y, X_tilde, w, b, delta)

    def post_selection_analysis(self, alpha=0.1, compute_intervals=False, gridrange=[-100, 100], tol_beta=1e-5,
                                tol_kkt=0.1):
        """
        implements the post selection analysis proposed by

        Lee, J. D., Sun, D. L., Sun, Y., & Taylor, J. E. (2016).
        Exact post-selection inference, with application to the lasso.
        The Annals of Statistics, 44(3), 907-927.


        See: https://github.com/selective-inference/Python-software

        :param float alpha: The (1-alpha)*100% selective confidence intervals
        :param bool UMAU: Whethter UMAU intervals should be calculated
        :return: DataFrame with one entry per active variable. Columns are

        'variable', 'pval', 'lasso', 'beta','Zscore', 'lower_ci', 'upper_ci', 'lower_trunc', 'upper_trunc', 'sd'.

        """
        if self.__SUX is None and self.__islmm:
            RuntimeError("The model has not been trained yet")

        X = self.__SUX if self.__islmm else self.__X
        y = self.__SUy[:, 0] if self.__islmm else self.__y[:,0]
        w = self.__w
        P = self.__P
        l2 = self.__l2
        l1 = self.__l1
        sigma = self.__sigma if self.__sigma is not None else np.std(y)

        # compute the hessian of the active set M
        # H(M) = (XT_M*X_M + l2*P)^-1
        H = (np.dot(np.transpose(X), X) + l2*P)

        # Check KKT condition
        g = (np.dot(H, w) - np.dot(np.transpose(X), y))/l1
        if np.any(np.fabs(g) > 1.+tol_kkt * np.sqrt(np.sum(np.power(y, 2)))):
            Warning("Beta does not satisfy the KKT conditions (to within specified tolerances)")

        active = np.where(np.fabs(w) > tol_beta / np.sqrt(np.sum(np.power(X, 2), axis=0)))[0]
        print("Active", active)
        active_signs = np.sign(w[active])

        if not active.size:
            Warning("Model is empty")
            return None

        if np.any(np.sign(g[active]) != active_signs):
            Warning(
                "Solution beta does not satisfy the KKT conditions (to within specified tolerances). " +
                "You might try rerunning GELMMnet with a lower setting of the 'thresh' parameter, " +
                "for a more accurate convergence."
            )

        XM = X[:, active]
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



