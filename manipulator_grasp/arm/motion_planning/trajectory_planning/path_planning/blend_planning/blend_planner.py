import copy
from typing import Union, Iterable, List, Tuple

import numpy as np

from ..path_parameter import PathParameter
from ..path_planner import PathPlanner


class BlendPlanner:
    def __init__(self, path_parameters: Union[List[PathParameter], Tuple[PathParameter]],
                 radii: Union[np.ndarray, Iterable, int, float]) -> None:
        super().__init__()
        self.path_parameters = copy.deepcopy(path_parameters)
        self.radii = copy.deepcopy(radii)
        self.s_array = []
        self.lengths_cumsum = []
        self.planners: List[PathPlanner] = []
        self.calculate_radii()
        self.plan()

    def calculate_radii(self):
        for i, radius in enumerate(self.radii):
            max_radius = min((self.path_parameters[i].get_length(), self.path_parameters[i + 1].get_length())) / 2.0

            if radius > max_radius:
                self.radii[i] = max_radius

    def plan(self) -> None:

        lengths = [0.0, 0.0]
        for i in range(len(self.path_parameters)):
            if i == 0:
                length = self.path_parameters[i].get_length()
                if len(self.radii) != 0:
                    length -= self.radii[i]
            elif i == len(self.path_parameters) - 1:
                length = self.path_parameters[i].get_length() - self.radii[i - 1]
            else:
                length = self.path_parameters[i].get_length() - self.radii[i - 1] - self.radii[i]

            lengths.append(length)

            if i != len(self.path_parameters) - 1:
                lengths.append(self.radii[i])
        lengths.append(0.0)

        self.lengths_cumsum = np.cumsum(lengths)

        self.planners = [PathPlanner(path_parameter) for path_parameter in self.path_parameters]

    def interpolate(self, s: float):

        sl = s * self.lengths_cumsum[-1]

        if sl <= 0.0:
            return self.planners[0].interpolate(0.0)
        elif sl >= self.lengths_cumsum[-1]:
            return self.planners[-1].interpolate(1.0)
        else:
            for i, ei in enumerate(self.lengths_cumsum[2:-1]):
                if sl > ei:
                    continue

                if i % 2:
                    ss = (sl - self.lengths_cumsum[i + 1]) / (
                            self.lengths_cumsum[i + 2] - self.lengths_cumsum[i + 1])
                    alpha = 6 * ss ** 5 - 15 * ss ** 4 + 10 * ss ** 3

                    s1 = (sl - self.lengths_cumsum[i - 1]) / (
                            self.lengths_cumsum[i + 2] - self.lengths_cumsum[i - 1])
                    s2 = (sl - self.lengths_cumsum[i + 1]) / (
                            self.lengths_cumsum[i + 4] - self.lengths_cumsum[i + 1])

                    return self.planners[i // 2].interpolate(s1) + alpha * (
                            self.planners[i // 2 + 1].interpolate(s2) - self.planners[i // 2].interpolate(s1))
                else:
                    ss = (sl - self.lengths_cumsum[i]) / (self.lengths_cumsum[i + 3] - self.lengths_cumsum[i])

                    return self.planners[i // 2].interpolate(ss)
