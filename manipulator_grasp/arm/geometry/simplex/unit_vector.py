from typing import overload, Union, Iterable

import numpy as np

from .point import Point
from .vector import Vector


class UnitVector(Vector):

    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self, t0: Union[np.ndarray, Iterable, int, float]):
        ...

    @overload
    def __init__(self, t0: Union[np.ndarray, Iterable, int, float], t1: Union[np.ndarray, Iterable, int, float]):
        ...

    @overload
    def __init__(self, t0: Point):
        ...

    @overload
    def __init__(self, t0: Point, t1: Point):
        ...

    def __init__(self, t0=None, t1=None) -> None:
        super().__init__(t0, t1)
        self.t = self.t / np.linalg.norm(self.t)
