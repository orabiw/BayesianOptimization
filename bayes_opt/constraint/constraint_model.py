""" `bayes_opt.constraint` """
# pylint: disable=invalid-name
import typing as t

import numpy as np

from bayes_opt import types
from bayes_opt.constraint import (
    base_constraint_model,
    single_constraint_model,
    multiple_constraint_model,
)


class ConstraintModel(
    base_constraint_model.BaseConstraintModel
):  # pylint:disable=abstract-method
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
        random_state: types.RandomState = None,
    ):
        if isinstance(lb, float):
            lb = np.array([lb])

        if len(lb) == 1:
            return single_constraint_model.SingleConstraintModel(
                fun, lb, ub, random_state
            )

        return multiple_constraint_model.MultipleConstraintModel(
            fun, lb, ub, random_state
        )
