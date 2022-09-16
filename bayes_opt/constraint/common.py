""" `bayes_opt.constraint.common` """
# pylint:disable=invalid-name
import numpy as np

import scipy.stats
import sklearn.gaussian_process

from bayes_opt import types


def predict(gp, X, lb, ub):
    """finds the p_lower and p_upper"""
    if lb == np.inf:
        return np.array([0]), np.array([1])

    y_mean, y_std = gp.predict(X, return_std=True)

    return (
        scipy.stats.norm(loc=y_mean, scale=y_std).cdf(boundary) for boundary in (lb, ub)
    )


def create_regressor(
    random_state: types.RandomState,
) -> sklearn.gaussian_process.GaussianProcessRegressor:
    """creates a `GaussianProcessRegressor`"""
    return sklearn.gaussian_process.GaussianProcessRegressor(
        kernel=sklearn.gaussian_process.kernels.Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )
