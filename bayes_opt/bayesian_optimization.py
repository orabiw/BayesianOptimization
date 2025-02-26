""" `bayes_opt.bayesian_optimization` """
import typing as t
import warnings
from queue import Queue, Empty

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import bayes_opt.constraint
import bayes_opt.event
import bayes_opt.util

from bayes_opt import domain_reduction, logger, target_space


class Observable:
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """

    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: {} for event in events}

    def get_subscribers(self, event):
        """get_subscribers"""
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        """subscribe"""
        if callback is None:
            callback = getattr(subscriber, "update")
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        """unsubscribe"""
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        """dispatch"""
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    """
    This class takes the function to optimize as well as the parameters bounds
    in order to find which values for the parameters yield the maximum value
    using bayesian optimization.

    Parameters
    ----------
    f: function
        Function to be maximized.

    pbounds: dict
        Dictionary with parameters names as keys and a tuple with minimum
        and maximum values.

    constraint: A ConstraintModel. Note that the names of arguments of the
        constraint function and of f need to be the same.

    random_state: int or numpy.random.RandomState, optional(default=None)
        If the value is an integer, it is used as the seed for creating a
        numpy.random.RandomState. Otherwise the random state provided is used.
        When set to None, an unseeded random state is generated.

    verbose: int, optional(default=2)
        The level of verbosity.

    bounds_transformer: DomainTransformer, optional(default=None)
        If provided, the transformation is applied to the bounds.

    Methods
    -------
    probe()
        Evaluates the function on the given points.
        Can be used to guide the optimizer.

    maximize()
        Tries to find the parameters that yield the maximum value for the
        given function.

    set_bounds()
        Allows changing the lower and upper searching bounds
    """

    _space: target_space.BaseTargetSpace

    def __init__(  # pylint:disable=too-many-arguments,no-member
        self,
        f: t.Callable,
        pbounds: t.Dict[str, t.Tuple[t.Union[int, float], t.Union[int, float]]],
        constraint: t.Optional[bayes_opt.constraint.ConstraintModel] = None,
        random_state: t.Optional[t.Union[int, np.random.RandomState]] = None,
        verbose: int = 2,
        bounds_transformer: t.Optional[domain_reduction.DomainTransformer] = None,
    ):
        self._random_state: np.random.RandomState = bayes_opt.util.ensure_rng(
            random_state
        )

        self._queue: Queue = Queue()

        # Internal GP regressor
        self._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=self._random_state,
        )

        if constraint is None:
            # Data structure containing the function to be optimized, the
            # bounds of its domain, and a record of the evaluations we have
            # done so far
            self._space = target_space.TargetSpace(f, pbounds, random_state)
            self.is_constrained = False
        else:
            constraint_ = bayes_opt.constraint.ConstraintModel(
                constraint.fun, constraint.lb, constraint.ub, random_state=random_state
            )

            self._space = target_space.ConstrainedTargetSpace(
                f, constraint_, pbounds, random_state
            )
            self.is_constrained = True

        self._verbose = verbose
        self._bounds_transformer = bounds_transformer

        if self._bounds_transformer:
            try:
                self._bounds_transformer.initialize(self._space)
            except (AttributeError, TypeError) as error:
                raise TypeError(
                    'The transformer must be an instance of "DomainTransformer"'
                ) from error

        super().__init__(events=bayes_opt.event.DEFAULT_EVENTS)

    @property
    def space(self) -> target_space.BaseTargetSpace:
        """space"""
        return self._space

    @property
    def constraint(self) -> t.Optional[bayes_opt.constraint.ConstraintModel]:
        """constraint"""
        if self.is_constrained and isinstance(
            self._space, target_space.ConstrainedTargetSpace
        ):
            return self._space.constraint

        return None

    @property
    def max(self) -> t.Dict[str, t.Any]:
        """max"""
        return self._space.max()

    @property
    def res(self) -> t.Dict[str, t.Any]:
        """res"""
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        self._space.register(params, target)
        self.dispatch(bayes_opt.event.Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """
        Evaluates the function on the given points. Useful to guide the optimizer.

        Parameters
        ----------
        params: dict or list
            The parameters where the optimizer will evaluate the function.

        lazy: bool, optional(default=True)
            If True, the optimizer will evaluate the points when calling
            maximize(). Otherwise it will evaluate it at the moment.
        """

        if lazy:
            self._queue.put(params)
        else:
            self._space.probe(params)
            self.dispatch(bayes_opt.event.Events.OPTIMIZATION_STEP)

    def suggest(self, utility_function):
        """Most promising point to probe next"""
        if len(self._space) == 0:
            return self._space.array_to_params(self._space.random_sample())

        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._gp.fit(self._space.params, self._space.target)
            if self.is_constrained:
                self.constraint.fit(self._space.params, self._space.constraint_values)

        # Finding argmax of the acquisition function.
        suggestion = bayes_opt.util.acq_max(
            ac=utility_function.utility,
            gp=self._gp,
            constraint=self.constraint,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state,
        )

        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty() and self._space.empty:
            init_points = max(init_points, 1)

        for _ in range(init_points):
            self._queue.put(self._space.random_sample())

    def _prime_subscriptions(self):
        subscriptions = (len(subs) for subs in self._events.values())

        if not any(subscriptions):
            _logger = logger.get_default_logger(self._verbose)
            self.subscribe(bayes_opt.event.Events.OPTIMIZATION_START, _logger)
            self.subscribe(bayes_opt.event.Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(bayes_opt.event.Events.OPTIMIZATION_END, _logger)

    def maximize(  # pylint:disable=too-many-arguments
        self,
        init_points=5,
        n_iter=25,
        acq="ucb",
        kappa=2.576,
        kappa_decay=1,
        kappa_decay_delay=0,
        xi=0.0,  # pylint:disable=invalid-name
        **gp_params
    ):
        """
        Probes the target space to find the parameters that yield the maximum
        value for the given function.

        Parameters
        ----------
        init_points : int, optional(default=5)
            Number of iterations before the explorations starts the exploration
            for the maximum.

        n_iter: int, optional(default=25)
            Number of iterations where the method attempts to find the maximum
            value.

        acq: {'ucb', 'ei', 'poi'}
            The acquisition method used.
                * 'ucb' stands for the Upper Confidence Bounds method
                * 'ei' is the Expected Improvement method
                * 'poi' is the Probability Of Improvement criterion.

        kappa: float, optional(default=2.576)
            Parameter to indicate how closed are the next parameters sampled.
                Higher value = favors spaces that are least explored.
                Lower value = favors spaces where the regression function is
                the highest.

        kappa_decay: float, optional(default=1)
            `kappa` is multiplied by this factor every iteration.

        kappa_decay_delay: int, optional(default=0)
            Number of iterations that must have passed before applying the
            decay to `kappa`.

        xi: float, optional(default=0.0)
            [unused]
        """
        self._prime_subscriptions()
        self.dispatch(bayes_opt.event.Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        self.set_gp_params(**gp_params)

        util = bayes_opt.util.UtilityFunction(
            kind=acq,
            kappa=kappa,
            xi=xi,
            kappa_decay=kappa_decay,
            kappa_decay_delay=kappa_decay_delay,
        )
        iteration = 0
        while not self._queue.empty() or iteration < n_iter:
            try:
                x_probe = self._queue.get(block=False)
            except Empty:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1
            self.probe(x_probe, lazy=False)

            if self._bounds_transformer and iteration > 0:
                # The bounds transformer should only modify the bounds after
                # the init_points points (only for the true iterations)
                self.set_bounds(self._bounds_transformer.transform(self._space))

        self.dispatch(bayes_opt.event.Events.OPTIMIZATION_END)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        """Set parameters to the internal Gaussian Process Regressor"""
        self._gp.set_params(**params)
