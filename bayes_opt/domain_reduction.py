""" `bayes_opt.domain_reduction` """
# pylint: disable=invalid-name
import typing as t

import numpy as np
import bayes_opt.target_space


def trim_bounds(
    bounds: np.ndarray,
    global_bounds: np.ndarray,
    minimum_window: t.Sequence[float],
) -> np.ndarray:
    """not the best name"""
    del global_bounds  # not used

    bounds = np.copy(bounds)

    # The below code block existed in the origianl code but it's not affecting any
    # variables as it changes the variables defined in the loop but noot the actual
    # bounds, keeping it here for reference.
    # for i, variable in enumerate(bounds):
    #     if variable[0] < global_bounds[i, 0]:
    #         variable[0] = global_bounds[i, 0]
    #     if variable[1] > global_bounds[i, 1]:
    #         variable[1] = global_bounds[i, 1]

    for i, variable in enumerate(bounds):
        lower, upper = variable

        window_width = abs(lower - upper)

        if lower > upper:
            lower, upper = upper, lower

        half_delta = (minimum_window[i] - window_width) / 2.0

        if window_width < minimum_window[i]:
            lower -= half_delta
            upper += half_delta

        bounds[i, 0] = lower
        bounds[i, 1] = upper

    return bounds


def _current_d(current_optimal, previous_optimal, r):
    return 2.0 * (current_optimal - previous_optimal) / r


class DomainTransformer(t.Protocol):
    """The base transformer class"""

    def initialize(self, target_space: bayes_opt.target_space.BaseTargetSpace):
        """initialize"""

    def transform(self, target_space: bayes_opt.target_space.BaseTargetSpace):
        """transform"""


class SequentialDomainReductionTransformer:
    # pylint:disable=too-many-instance-attributes
    """
    A sequential domain reduction transformer bassed on the work by Stander, N. and
    Craig, K: "On the robustness of a simple domain reduction scheme for
    simulation‐based optimization"
    """

    minimum_window: t.Sequence
    original_bounds: np.ndarray
    bounds: t.List[np.ndarray]
    previous_optimal: np.ndarray
    current_optimal: np.ndarray
    bounds_range: np.ndarray
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
            if len(self.minimum_window_value) != len(target_space.bounds):
                raise ValueError(
                    "`minimum_window_value` does not equal `target_space.bounds`"
                )

            self.minimum_window = self.minimum_window_value
        else:
            self.minimum_window = [self.minimum_window_value] * len(target_space.bounds)

        optimal_bounds = np.mean(target_space.bounds, axis=1)
        self.previous_optimal = optimal_bounds
        self.current_optimal = optimal_bounds

        self.bounds_range = target_space.bounds[:, 1] - target_space.bounds[:, 0]

        d_value = _current_d(
            self.current_optimal, self.previous_optimal, self.bounds_range
        )

        self.previous_d = d_value
        self.current_d = d_value

        self._calc_values()

    def _calc_values(self):
        self.c = self.current_d * self.previous_d
        self.c_hat = np.sqrt(np.abs(self.c)) * np.sign(self.c)

        self.gamma = 0.5 * (
            self.gamma_pan * (1.0 + self.c_hat) + self.gamma_osc * (1.0 - self.c_hat)
        )

        self.contraction_rate = self.eta + np.abs(self.current_d) * (
            self.gamma - self.eta
        )

        self.bounds_range = self.contraction_rate * self.bounds_range

    def transform(self, target_space: bayes_opt.target_space.BaseTargetSpace) -> dict:
        # setting the previous
        self.previous_optimal = self.current_optimal
        self.previous_d = self.current_d

        self.current_optimal = target_space.params[np.argmax(target_space.target)]
        self.current_d = _current_d(
            self.current_optimal, self.previous_optimal, self.bounds_range
        )

        self._calc_values()

        half_r = 0.5 * self.bounds_range
        bounds = np.array(
            [
                self.current_optimal - half_r,
                self.current_optimal + half_r,
            ]
        ).T

        bounds = trim_bounds(bounds, self.original_bounds, self.minimum_window)
        self.bounds.append(bounds)

        return {param: bounds[i, :] for i, param in enumerate(target_space.keys)}
