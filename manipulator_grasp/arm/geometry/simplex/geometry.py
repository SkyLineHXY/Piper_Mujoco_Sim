import copy
from abc import ABC
from typing import overload

import numpy as np

from manipulator_grasp.arm.constanst import MathConst


class Geometry(ABC):

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, t) -> None:
        ...

    def __init__(self, t=None) -> None:
        super().__init__()
        self.t = None

    def __add__(self, other):
        return self.__class__(self.t + other.get_t())

    def __sub__(self, other):
        return self.__class__(self.t - other.get_t())

    def __mul__(self, other):
        return self.__class__(other * self.t)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1.0 / other)

    def __str__(self) -> str:
        return str(self.t)

    def __eq__(self, other):
        # return np.array_equal(self.t, other.get_t())
        return np.linalg.norm(self.t - other.get_t()) < MathConst.EPS

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.t)

    def get_t(self) -> np.ndarray:
        return copy.deepcopy(self.t)

    def get_t_3d(self) -> np.ndarray:
        t = self.get_t()
        if t.size == 2:
            return np.append(t, 0.0)
        elif t.size == 3:
            return t
