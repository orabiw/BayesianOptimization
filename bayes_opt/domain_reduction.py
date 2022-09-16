""" `bayes_opt.domain_reduction` """
# pylint: disable=invalid-name
import abc
import typing as t

import numpy as np
import bayes_opt.target_space


class DomainTransformer(abc.ABC):
    """The base transformer class"""

    @abc.abstractmethod
    def initialize(self, target_space: bayes_opt.target_space.BaseTargetSpace):
        """initialize"""
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, target_space: bayes_opt.target_space.BaseTargetSpace):
        """transform"""
        raise NotImplementedError


class SequentialDomainReductionTransformer(DomainTransformer):
    # pylint:disable=too-many-instance-attributes
    """
    A sequential domain reduction transformer bassed on the work by Stander, N. and Craig, K:
    "On the robustness of a simple domain reduction scheme for simulation‐based optimization"
    """

    minimum_window: t.Sequence
    original_bounds: np.ndarray
    bounds: t.List[np.ndarray]
    previous_optimal: np.ndarray
    current_optimal: np.ndarray
    r: np.ndarray
    previous_d: np.ndarray
    current_d: np.ndarray
    c: np.ndarray
    c_hat: np.ndarray
    gamma: np.ndarray
    contraction_rate: np.ndarray

    def __init__(
        self,
        gamma_osc: float = 0.7,
        gamma_pan: float = 1.0,
        eta: float = 0.9,
        minimum_window: t.Optional[t.Union[t.List[t.Optional[float]], float]] = 0.0,
    ) -> None:
        self.gamma_osc = gamma_osc
        self.gamma_pan = gamma_pan
        self.eta = eta
        self.minimum_window_value = minimum_window

    def initialize(self, target_space: bayes_opt.target_space.BaseTargetSpace) -> None:
        """Initialize all of the parameters"""
        self.original_bounds = np.copy(target_space.bounds)
        self.bounds = [self.original_bounds]

        # Set the minimum window to an array of length bounds
        if isinstance(self.minimum_window_value, (list, np.ndarray)):
            assert len(self.minimum_window_value) == len(target_space.bounds)
            self.minimum_window = self.minimum_window_value
        else:
            self.minimum_window = [self.minimum_window_value] * len(target_space.bounds)

        self.previous_optimal = np.mean(target_space.bounds, axis=1)
        self.current_optimal = np.mean(target_space.bounds, axis=1)
        self.r = target_space.bounds[:, 1] - target_space.bounds[:, 0]

        self.previous_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r

        self.current_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (
            self.gamma_pan * (1.0 + self.c_hat) + self.gamma_osc * (1.0 - self.c_hat)
        )

        self.contraction_rate = self.eta + np.abs(self.current_d) * (
            self.gamma - self.eta
        )

        self.r = self.contraction_rate * self.r

    def _update(self, target_space: bayes_opt.target_space.BaseTargetSpace) -> None:

        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        self.current_optimal = target_space.params[np.argmax(target_space.target)]

        self.current_d = 2.0 * (self.current_optimal - self.previous_optimal) / self.r

        self.c = self.current_d * self.previous_d

        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (
            self.gamma_pan * (1.0 + self.c_hat) + self.gamma_osc * (1.0 - self.c_hat)
        )

        self.contraction_rate = self.eta + np.abs(self.current_d) * (
            self.gamma - self.eta
        )

        self.r = self.contraction_rate * self.r

    def _trim(self, new_bounds: np.ndarray, global_bounds: np.ndarray) -> np.ndarray:
        for i, variable in enumerate(new_bounds):
            if variable[0] < global_bounds[i, 0]:
                variable[0] = global_bounds[i, 0]
            if variable[1] > global_bounds[i, 1]:
                variable[1] = global_bounds[i, 1]
        for i, entry in enumerate(new_bounds):
            if entry[0] > entry[1]:
                new_bounds[i, 0] = entry[1]
                new_bounds[i, 1] = entry[0]
            window_width = abs(entry[0] - entry[1])
            if window_width < self.minimum_window[i]:
                new_bounds[i, 0] -= (self.minimum_window[i] - window_width) / 2.0
                new_bounds[i, 1] += (self.minimum_window[i] - window_width) / 2.0

        return new_bounds

    def _create_bounds(self, parameters: dict, bounds: np.ndarray) -> dict:
        return {param: bounds[i, :] for i, param in enumerate(parameters)}

    def transform(self, target_space: bayes_opt.target_space.BaseTargetSpace) -> dict:

        self._update(target_space)

        new_bounds = np.array(
            [self.current_optimal - 0.5 * self.r, self.current_optimal + 0.5 * self.r]
        ).T

        self._trim(new_bounds, self.original_bounds)
        self.bounds.append(new_bounds)
        return self._create_bounds(target_space.keys, new_bounds)
