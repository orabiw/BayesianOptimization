"""
observers...
"""
from __future__ import annotations

import abc
from datetime import datetime
import typing as t

import bayes_opt.event

if t.TYPE_CHECKING:
    import bayes_opt.bayesian_optimization


class Observer(abc.ABC):  # pylint:disable=too-few-public-methods
    """`Observer` class"""

    @abc.abstractmethod
    def update(
        self,
        event: bayes_opt.event.Events,
        instance: bayes_opt.bayesian_optimization.BayesianOptimization,
    ):
        """`Observer.update`"""
        raise NotImplementedError


class Tracker:
    """Tracker"""

    _previous_max: t.Optional[t.Dict[str, t.Any]] = None
    _previous_max_params: t.Optional[t.Dict[str, t.Any]] = None
    _start_time: t.Optional[datetime] = None
    _previous_time: t.Optional[datetime] = None
    _iterations: int = 0

    def update_tracker(
        self,
        event: bayes_opt.event.Events,
        instance: bayes_opt.bayesian_optimization.BayesianOptimization,
    ) -> None:
        """update_tracker"""
        if event != bayes_opt.event.Events.OPTIMIZATION_STEP:
            return

        self._iterations += 1
        current_max = instance.max

        if self._previous_max is None or current_max["target"] > self._previous_max:
            self._previous_max = current_max["target"]
            self._previous_max_params = current_max["params"]

    def time_metrics(self) -> t.Tuple[str, float, float]:
        """time_metrics"""
        now = datetime.now()

        if self._start_time is None:
            self._start_time = now

        if self._previous_time is None:
            self._previous_time = now

        time_elapsed = now - self._start_time
        time_delta = now - self._previous_time

        self._previous_time = now

        return (
            now.strftime("%Y-%m-%d %H:%M:%S"),
            time_elapsed.total_seconds(),
            time_delta.total_seconds(),
        )
