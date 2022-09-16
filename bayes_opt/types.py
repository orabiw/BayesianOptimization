""" `bayes_opt.types` """
import typing as t

import numpy as np

RandomState = t.Union[int, np.random.RandomState, None]  # pylint:disable=no-member
