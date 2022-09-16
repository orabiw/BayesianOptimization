""" `bayes_opt.logger` """
# pylint:disable=invalid-name
from __future__ import print_function
import os
import json

import bayes_opt.observer
import bayes_opt.event
import bayes_opt.util


def get_default_logger(verbose):
    """get_default_logger"""
    return ScreenLogger(verbose=verbose)


class ScreenLogger(bayes_opt.observer.Tracker):
    """ScreenLogger"""

    _default_cell_size = 9
    _default_precision = 4

    def __init__(self, verbose=2):
        self._verbose = verbose
        self._header_length = None
        super().__init__()

    @property
    def verbose(self):
        """verbose"""
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = v

    def _format_number(self, x):
        if isinstance(x, int):
            s = f"{x:<{self._default_cell_size}}"
        else:
            s = f"{x:<{self._default_cell_size}.{self._default_precision}}"

        if len(s) > self._default_cell_size:
            if "." in s:
                return s[: self._default_cell_size]
            return s[: self._default_cell_size - 3] + "..."

        return s

    def _format_key(self, key):
        s = f"{key:^{self._default_cell_size}}"

        if len(s) > self._default_cell_size:
            return s[: self._default_cell_size - 3] + "..."

        return s

    def _step(self, instance, colour=bayes_opt.util.Colours.black):
        res = instance.res[-1]
        cells = []

        cells.append(self._format_number(self._iterations + 1))
        cells.append(self._format_number(res["target"]))

        for key in instance.space.keys:
            cells.append(self._format_number(res["params"][key]))

        return "| " + " | ".join(map(colour, cells)) + " |"

    def _header(self, instance):
        cells = []
        cells.append(self._format_key("iter"))
        cells.append(self._format_key("target"))
        for key in instance.space.keys:
            cells.append(self._format_key(key))

        line = "| " + " | ".join(cells) + " |"
        self._header_length = len(line)
        return line + "\n" + ("-" * self._header_length)

    def _is_new_max(self, instance):
        if instance.max["target"] is None:
            # During constrained optimization, there might not be a maximum
            # value since the optimizer might've not encountered any points
            # that fulfill the constraints.
            return False
        if self._previous_max is None:
            self._previous_max = instance.max["target"]
        return instance.max["target"] > self._previous_max

    def update(self, event, instance):
        """update"""
        if event == bayes_opt.event.Events.OPTIMIZATION_START:
            line = self._header(instance) + "\n"
        elif event == bayes_opt.event.Events.OPTIMIZATION_STEP:
            is_new_max = self._is_new_max(instance)
            if self._verbose == 1 and not is_new_max:
                line = ""
            else:
                if is_new_max:
                    colour = bayes_opt.util.Colours.purple
                else:
                    colour = bayes_opt.util.Colours.black

                line = self._step(instance, colour=colour) + "\n"
        elif event == bayes_opt.event.Events.OPTIMIZATION_END:
            line = "=" * self._header_length + "\n"

        if self._verbose:
            print(line, end="")

        self.update_tracker(event, instance)


class JSONLogger(bayes_opt.observer.Tracker):  # pylint:disable=too-few-public-methods
    """JSONLogger"""

    def __init__(self, path, reset=True):
        self._path = path if path[-5:] == ".json" else path + ".json"
        if reset:
            try:
                os.remove(self._path)
            except OSError:
                pass

        super().__init__()

    def update(self, event, instance):
        """update"""
        if event == bayes_opt.event.Events.OPTIMIZATION_STEP:
            data = dict(instance.res[-1])

            now, time_elapsed, time_delta = self.time_metrics()
            data["datetime"] = {
                "datetime": now,
                "elapsed": time_elapsed,
                "delta": time_delta,
            }

            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")

        self.update_tracker(event, instance)
