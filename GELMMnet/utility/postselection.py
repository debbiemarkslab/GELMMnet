import functools
import numpy as np
from scipy.stats import norm


def _min(vector):
    try:
        return min(vector)
    except ValueError:
        return np.inf


def _max(vector):
    try:
        return max(vector)
    except ValueError:
        return -np.inf


def _truncatednorm_surv(z, mean, a, b, sd):
    """
    Truncated normal survival function

    :return: returns P(Z > z)
    """
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


def _grid_search(grid, fun, val1, val2, gridpts=100, griddepth=1):
    """
    First grid search for CI calculation

    """
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

    return [lo, hi]


def _grid_bsearch(l, r, fun, val, gridpts=100, griddepth=0, below=True):
    """
    Second level grid binsearch for CI calculation

    """
    left = l
    right = r

    n = gridpts
    depth = 0
    while depth <= griddepth:
        grid = np.linspace(left, right, num=n)
        vals = np.array(list(map(fun, grid)))

        if below:
            ii = np.where(vals >= val)[0]

            if not ii.size:
                return grid[-1]

            i0 = int(np.min(ii))
            if not i0:
                return grid[0]
            left = grid[i0 - 1]
            right = grid[i0]

        else:
            ii = np.where(vals <= val)[0]

            if not ii.size:
                return grid[0]

            i0 = int(np.max(ii))
            if i0 >= (n - 1):
                return grid[-1]

            left = grid[i0]
            right = grid[i0+1]
        depth += 1

    return left if below else right


def _tg_limits(Z, A, b, eta, Sigma):
    """
    Compute the truncation interval and SD of the corresponding Gaussian

    :return: lower-bound, upper-bound, sd, estimate
    """
    target_estimate = np.sum(eta * Z)

    n = len(Z)
    eta_col = np.reshape(eta, (1, n))
    eta_row = np.reshape(eta, (n, 1))
    b = np.reshape(b, (len(b), 1))
    var_estimate = np.sum(np.dot(eta_col, np.dot(Sigma, eta_row)))
    cross_cov = np.dot(Sigma, eta_row)
    resid = np.dot(np.identity(n) - np.dot(np.reshape(cross_cov / var_estimate, (n, 1)), eta_col), Z)
    rho = np.dot(A, cross_cov / var_estimate)
    vec = (b - np.dot(A, np.reshape(resid, (len(resid),1)))) / rho
    vlo = _max(vec[rho > 0])
    vup = _min(vec[rho < 0])

    sd = np.sqrt(var_estimate)

    return vlo, vup, sd, target_estimate


def _tg_pval(estimate, vlo, vup, sd, null_value=0):
    """
    calculates p-value based on defined polyhedral

    :return: p-value
    """
    return _truncatednorm_surv(estimate, null_value, vlo, vup, sd)


def _tg_interval(estimate, vlo, vup, sd, alpha, gridrange=[-100, 100], gridpts=100, griddepth=1, flip=False):
    """
    Selection corrected CI calculation

    :return: (CI_lower,CI_upper),(tail_lower, tail_upper)
    """
    xg = np.linspace(gridrange[0]*sd, gridrange[1]*sd, num=gridpts)

    fun = functools.partial(lambda z, vlo, vup, sd, x: _truncatednorm_surv(z, x, vlo, vup, sd), estimate, vlo, vup, sd)
    int = _grid_search(xg, fun, alpha/2., 1 - alpha/2., gridpts, griddepth)
    tailarea = [fun(int[0]), 1 - fun(int[1])]

    if flip:
        return [-x for x in int[::-1]], tailarea[::-1]
    return int, tailarea