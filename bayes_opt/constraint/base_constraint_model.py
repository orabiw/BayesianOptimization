""" base constraint model """
# pylint:disable=invalid-name
import typing as t

import numpy as np


from bayes_opt import types


class BaseConstraintModel:
    """Base class for ConstraintModel classes"""

    def __init__(  # pylint:disable=unused-argument
        self,
        fun: t.Callable,
        lb: t.Union[float, np.ndarray],
        ub: t.Union[float, np.ndarray],
        random_state: types.RandomState = None,
    ):
        self.fun: t.Callable = fun
        self.lb: np.ndarray = np.array([lb]) if isinstance(lb, float) else lb
        self.ub: np.ndarray = np.array([ub]) if isinstance(ub, float) else ub
        self.no_features = None

    def eval(self, **kwargs):
        """
        Evaluates the constraint function.
        """
        try:
            return self.fun(**kwargs)
        except TypeError as e:
            msg = (
                "Encountered TypeError when evaluating constraint "
                "function. This could be because your constraint function "
                "doesn't use the same keyword arguments as the target "
                f"function. Original error message:\n\n{e}"
            )

            e.args = (msg,)
            raise

    def fit(self, X, Y):
        """
        Fits internal GaussianProcessRegressor's to the data.
        """
        raise NotImplementedError()

    def predict(self, X):
        """
        Returns the probability that the constraint is fulfilled at `X` based
        on the internal Gaussian Process Regressors.

        Note that this does not try to approximate the values of the constraint
        function, but probability that the constraint function is fulfilled.
        For the former, see `ConstraintModel.approx()`.
        """
        raise NotImplementedError()

    def approx(self, X):
        """
        Returns the approximation of the constraint function using the internal
        Gaussian Process Regressors.
        """
        raise NotImplementedError()

    def allowed(self, constraint_values):
        """
        Checks whether `constraint_values` are below the specified limits.
        """
        raise NotImplementedError()
