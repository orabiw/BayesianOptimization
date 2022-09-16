""" `bayes_opt.constraint` """
# pylint: disable=invalid-name
import typing as t

import numpy as np

import scipy.stats
from sklearn import gaussian_process

RandomState = t.Union[int, np.random.RandomState, None]  # pylint:disable=no-member


def _predict(gp, X, lb, ub):
    if lb == np.inf:
        return np.array([0]), np.array([1])

    y_mean, y_std = gp.predict(X, return_std=True)

    return (
        scipy.stats.norm(loc=y_mean, scale=y_std).cdf(boundary) for boundary in (lb, ub)
    )


def _create_regressor(
    random_state: RandomState,
) -> gaussian_process.GaussianProcessRegressor:
    return gaussian_process.GaussianProcessRegressor(
        kernel=gaussian_process.kernels.Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )


class BaseConstraintModel:
    """Base class for ConstraintModel classes"""

    def __init__(  # pylint:disable=unused-argument
        self,
        fun: t.Callable,
        lb: t.Union[float, np.ndarray],
        ub: t.Union[float, np.ndarray],
        random_state: RandomState = None,
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


class SingleConstraintModel(BaseConstraintModel):
    """Single model constraint"""

    def __init__(
        self,
        fun: t.Callable,
        lb: t.Union[float, np.ndarray],
        ub: t.Union[float, np.ndarray],
        random_state: RandomState = None,
    ):
        super().__init__(fun, lb, ub, random_state)
        self.model = _create_regressor(random_state)

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

        p_lower, p_upper = _predict(self.model, X, self.lb[0], self.ub[0])
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


class MultipleConstraintModel(BaseConstraintModel):
    """Multiple Constraint Model"""

    def __init__(
        self,
        fun: t.Callable,
        lb: t.Union[float, np.ndarray],
        ub: t.Union[float, np.ndarray],
        random_state: RandomState = None,
    ):
        super().__init__(fun, lb, ub, random_state)
        self.model_size = len(self.lb)
        self.model = [_create_regressor(random_state) for _ in range(self.model_size)]

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
            p_lower, p_upper = _predict(gp, X, self.lb[j], self.ub[j])
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


class ConstraintModel(BaseConstraintModel):  # pylint:disable=abstract-method
    """
    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    fun: function
        Constraint function. If multiple constraints are handled, this should
        return a numpy.ndarray of appropriate size.

    lb: numeric or numpy.ndarray
        Upper limit(s) for the constraints. The return value of `fun` should
        have exactly this shape.

    ub: numeric or numpy.ndarray
        Upper limit(s) for the constraints. The return value of `fun` should
        have exactly this shape.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provided is used.
        When set to None, an unseeded random state is generated.

    Note
    ----
    In case of multiple constraints, this model assumes conditional
    independence. This means that for each constraint, the probability of
    fulfillment is the cdf of a univariate Gaussian. The overall probability
    is a simply the product of the individual probabilities.
    """

    def __new__(
        cls,
        fun: t.Callable,
        lb: t.Union[float, np.ndarray],
        ub: t.Union[float, np.ndarray],
        random_state: RandomState = None,
    ):
        if isinstance(lb, float):
            lb = np.array([lb])

        if len(lb) == 1:
            return SingleConstraintModel(fun, lb, ub, random_state)

        return MultipleConstraintModel(fun, lb, ub, random_state)
