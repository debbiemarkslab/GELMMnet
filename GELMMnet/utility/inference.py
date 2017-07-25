import numpy as np
from numba import jit


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
def _optimize_gelnet(y, X, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, with_intercept):
    obj_old = _calc_glnet_obj(S, y, Pw, w, l1, l2, n, m)

    # start optimization
    # optimize for max_iter steps
    for i in range(max_iter):
        # update each weight individually
        for j in range(m):
            w_old = w[j]

            w[j] = _update_wj(X, y, P, w, l1, l2, S, Pw, n, m, j)
            wj_dif = w[j] - w_old

            # update fit
            if np.not_equal(wj_dif, 0):
                S += X[:, j] * wj_dif  # dont know if that is correct
                Pw += P[:, j] * wj_dif

        # update bias
        b_diff = 0
        if with_intercept:
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


@jit(nopython=True)
def _predict(X, y, X_tilde, w, b, delta):

    n_test = X_tilde.shape[0]
    n_train = X.shape[0]
    idx = w.nonzero()[0]
    Xtmp = np.concatenate((X, X_tilde), axis=0)

    # calculate Covariance matrices
    K = 1. / Xtmp.shape[0] * np.dot(Xtmp, Xtmp.T)
    idx_tt = np.arange(n_train)
    idx_vt = np.arange(n_train, n_test + n_train)

    K_tt = K[idx_tt][:, idx_tt]
    K_vt = K[idx_vt][:, idx_tt]

    if idx.shape[0] == 0:
        return np.dot(K_vt, np.linalg.solve(K_tt + delta * np.eye(n_train), y))

    return np.dot(X_tilde[:, idx], w[idx]) + b + np.dot(K_vt, np.linalg.solve(K_tt + delta * np.eye(n_train),
                                                                             y[:, 0] - np.dot(X[:, idx],
                                                                                            w[idx]) + b))

@jit(nonpython=True)
def max_l1(y, X):
    """
    returns the upper limit of l1 (i.e., smallest value that yields a model with all zero weights)

    """
    b = np.mean(y)
    xy = np.mean(np.dot(X.T, (y - b)), axis=1)
    return np.max(np.fabs(xy))


@jit(nonpython=True)
def _mse(y, ypred):
    """
    Calculates the mean squered error
    :param y:
    :param ypred:
    :return:
    """
    return np.nanmean(np.power((y - ypred), 2))


@jit(nonpython=True)
def _parameter_search(fold, alpha, ratio, w, b, delta, isIntercept, ytrain, Xtrain, ytest, Xtest, P, eps, max_iter):
    """
    Function for grid search evaluation

    :param fold:
    :param alpha:
    :param ratio:
    :param metric:
    :param w:
    :param b:
    :param delta:
    :param ytrain:
    :param Xtrain:
    :param ytest:
    :param Xtest:
    :param P:
    :param eps:
    :param max_iter:
    :return: error,l1,l2
    """

    n,m = Xtrain.shape

    # initial parameter estimates
    S = np.dot(Xtrain, w) + b
    Pw = np.dot(P, w)

    l1 = ratio * alpha
    l2 = alpha * (1. - ratio)

    w, b = _optimize_gelnet(ytrain, Xtrain, P, l1, l2, S, Pw, n, m, max_iter, eps, w, b, isIntercept)

    ypred = _predict(Xtrain, ytrain, Xtest, w, b, delta)

    return fold, _mse(ytest, ypred), l1, l2