""" multiple constraint model """
# pylint: disable=invalid-name
import typing as t

import numpy as np


from bayes_opt import types
from bayes_opt.constraint import common, base_constraint_model


class MultipleConstraintModel(base_constraint_model.BaseConstraintModel):
    """Multiple Constraint Model"""

    def __init__(
        self,
        fun: t.Callable,
        lb: t.Union[float, np.ndarray],
        ub: t.Union[float, np.ndarray],
        random_state: types.RandomState = None,
    ):
        super().__init__(fun, lb, ub, random_state)
        self.model_size = len(self.lb)
        self.model = [
            common.create_regressor(random_state) for _ in range(self.model_size)
        ]

    def fit(self, X, Y):
        """
        Fits internal GaussianProcessRegressor's to the data.
        """
        for i, gp in enumerate(self.model):
            gp.fit(X, Y[:, i])

        self.no_features = self.model[0].n_features_in_

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

        result = np.ones(X.shape[0])
        for j, gp in enumerate(self.model):
            p_lower, p_upper = common.predict(gp, X, self.lb[j], self.ub[j])
            result = result * (p_upper - p_lower)

        return result.reshape(X_shape[:-1])

    def approx(self, X):
        """
        Returns the approximation of the constraint function using the internal
        Gaussian Process Regressors.
        """
        X_shape = X.shape
        X = X.reshape((-1, self.no_features))

        result = np.column_stack([gp.predict(X) for gp in self.model])
        return result.reshape(X_shape[:-1] + (self.model_size,))

    def allowed(self, constraint_values):
        """
        Checks whether `constraint_values` are below the specified limits.
        """
        return np.all(constraint_values <= self.ub, axis=-1) & np.all(
            constraint_values >= self.lb, axis=-1
        )
