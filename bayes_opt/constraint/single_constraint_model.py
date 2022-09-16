""" single constraint model """
# pylint: disable=invalid-name
import typing as t

import numpy as np


from bayes_opt import types
from bayes_opt.constraint import common, base_constraint_model


class SingleConstraintModel(base_constraint_model.BaseConstraintModel):
    """Single model constraint"""

    def __init__(
        self,
        fun: t.Callable,
        lb: t.Union[float, np.ndarray],
        ub: t.Union[float, np.ndarray],
        random_state: types.RandomState = None,
    ):
        super().__init__(fun, lb, ub, random_state)
        self.model = common.create_regressor(random_state)

    def fit(self, X, Y):
        """
        Fits internal GaussianProcessRegressor's to the data.
        """
        self.model.fit(X, Y)
        self.no_features = self.model.n_features_in_

    def predict(self, X):
        """
        Returns the probability that the constraint is fulfilled at `X` based
        on the internal Gaussian Process Regressors.

        Note that this does not try to approximate the values of the constraint
        function, but probability that the constraint function is fulfilled.
        For the former, see `ConstraintModel.approx()`.
        """
        X_shape = X.shape
        X = X.reshape((-1, self.no_features))

        p_lower, p_upper = common.predict(self.model, X, self.lb[0], self.ub[0])
        result = p_upper - p_lower
        return result.reshape(X_shape[:-1])

    def approx(self, X):
        """
        Returns the approximation of the constraint function using the internal
        Gaussian Process Regressors.
        """
        X_shape = X.shape
        X = X.reshape((-1, self.no_features))

        return self.model.predict(X).reshape(X_shape[:-1])

    def allowed(self, constraint_values):
        """
        Checks whether `constraint_values` are below the specified limits.
        """
        return np.less_equal(self.lb, constraint_values) & np.less_equal(
            constraint_values, self.ub
        )
