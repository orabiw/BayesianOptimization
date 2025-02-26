""" `bayes_opt.util` """
# pylint:disable=invalid-name
import json
import typing as t
import warnings

import numpy as np
import scipy.stats
import scipy.optimize


def acq_max(  # pylint:disable=too-many-arguments,too-many-locals
    ac,
    gp,
    y_max,
    bounds,
    random_state,
    constraint=None,
    n_warmup=10000,
    n_iter=10,
):
    """
    A function to find the maximum of the acquisition function

    It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
    optimization method. First by sampling `n_warmup` (1e5) points at random,
    and then running L-BFGS-B from `n_iter` (250) random starting points.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.

    :param random_state:
        instance of np.RandomState random number generator

    :param constraint:
        A ConstraintModel.

    :param n_warmup:
        number of times to randomly sample the acquisition function

    :param n_iter:
        number of times to run scipy.minimize

    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Warm up with random points
    x_tries = random_state.uniform(
        bounds[:, 0], bounds[:, 1], size=(n_warmup, bounds.shape[0])
    )
    ys = ac(x_tries, gp=gp, y_max=y_max)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    # Explore the parameter space more throughly
    x_seeds = random_state.uniform(
        bounds[:, 0], bounds[:, 1], size=(n_iter, bounds.shape[0])
    )

    if constraint is not None:

        def to_minimize(x):
            target = -ac(x.reshape(1, -1), gp=gp, y_max=y_max)
            p_constraint = constraint.predict(x.reshape(1, -1))

            # TODO: This is not exactly how Gardner et al do it.
            # Their way would require the result of the acquisition function
            # to be strictly positive (or negative), which is not the case
            # here. For a negative target value, we use Gardner's version. If
            # the target is positive, we instead slightly rescale the target
            # depending on the probability estimate to fulfill the constraint.
            if target < 0:
                return target * p_constraint

            return target / (0.5 + p_constraint)

    else:

        def to_minimize(x):
            return -ac(x.reshape(1, -1), gp=gp, y_max=y_max)

    for x_try in x_seeds:
        # Find the minimum of minus the acquisition function
        res = scipy.optimize.minimize(
            to_minimize, x_try, bounds=bounds, method="L-BFGS-B"
        )

        # See if success
        if not res.success:
            continue

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -np.squeeze(res.fun) >= max_acq:
            x_max = res.x
            max_acq = -np.squeeze(res.fun)

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])


class UtilityFunction:
    """
    An object to compute the acquisition functions.
    """

    def __init__(  # pylint:disable=too-many-arguments
        self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0
    ):

        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0

        if kind not in ["ucb", "ei", "poi"]:
            err = (
                "The utility function "
                f"{kind} has not been implemented, "
                "please choose one of ucb, ei, or poi."
            )

            raise NotImplementedError(err)

        self.kind = kind

    def update_params(self):
        """update_params"""
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_max):
        """utility"""
        if self.kind == "ucb":
            return self._ucb(x, gp, self.kappa)
        if self.kind == "ei":
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == "poi":
            return self._poi(x, gp, y_max, self.xi)

        raise ValueError("Unknown value for kind")

    @staticmethod
    def _ucb(x, gp, kappa):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        return mean + kappa * std

    @staticmethod
    def _ei(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = mean - y_max - xi
        z = a / std
        return a * scipy.stats.norm.cdf(z) + std * scipy.stats.norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        z = (mean - y_max - xi) / std
        return scipy.stats.norm.cdf(z)


def load_logs(optimizer, logs):
    """Load previous ..."""

    if isinstance(logs, str):
        logs = [logs]

    for log in logs:
        with open(log, "r", encoding="utf-8") as j:
            while True:
                try:
                    iteration = next(j)
                except StopIteration:
                    break

                iteration = json.loads(iteration)
                try:
                    optimizer.register(
                        params=iteration["params"],
                        target=iteration["target"],
                    )
                except KeyError:
                    pass

    return optimizer


def ensure_rng(  # pylint:disable=no-member
    random_state: t.Optional[t.Union[int, np.random.RandomState]] = None
) -> np.random.RandomState:
    """
    Creates a random number generator based on an optional seed.  This can be
    an integer or another random state for a seeded rng, or None for an
    unseeded rng.
    """
    if isinstance(random_state, np.random.RandomState):
        return random_state

    if random_state is None or isinstance(random_state, int):
        return np.random.RandomState(random_state)

    raise ValueError("Invalid random_state")


class Colours:
    """Print in nice colours."""

    BLUE = "\033[94m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    END = "\033[0m"
    GREEN = "\033[92m"
    PURPLE = "\033[95m"
    RED = "\033[91m"
    UNDERLINE = "\033[4m"
    YELLOW = "\033[93m"

    @classmethod
    def _wrap_colour(cls, s, colour):
        return colour + s + cls.END

    @classmethod
    def black(cls, s):
        """Wrap text in black."""
        return cls._wrap_colour(s, cls.END)

    @classmethod
    def blue(cls, s):
        """Wrap text in blue."""
        return cls._wrap_colour(s, cls.BLUE)

    @classmethod
    def bold(cls, s):
        """Wrap text in bold."""
        return cls._wrap_colour(s, cls.BOLD)

    @classmethod
    def cyan(cls, s):
        """Wrap text in cyan."""
        return cls._wrap_colour(s, cls.CYAN)

    @classmethod
    def darkcyan(cls, s):
        """Wrap text in darkcyan."""
        return cls._wrap_colour(s, cls.DARKCYAN)

    @classmethod
    def green(cls, s):
        """Wrap text in green."""
        return cls._wrap_colour(s, cls.GREEN)

    @classmethod
    def purple(cls, s):
        """Wrap text in purple."""
        return cls._wrap_colour(s, cls.PURPLE)

    @classmethod
    def red(cls, s):
        """Wrap text in red."""
        return cls._wrap_colour(s, cls.RED)

    @classmethod
    def underline(cls, s):
        """Wrap text in underline."""
        return cls._wrap_colour(s, cls.UNDERLINE)

    @classmethod
    def yellow(cls, s):
        """Wrap text in yellow."""
        return cls._wrap_colour(s, cls.YELLOW)
