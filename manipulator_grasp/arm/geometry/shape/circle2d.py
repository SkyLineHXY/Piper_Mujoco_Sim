import copy
from typing import overload, Union, Iterable

import numpy as np

from ..simplex import Point


class Circle2D:

    @overload
    def __init__(self):
        ...

    @overload
    def __init__(self, center: Union[np.ndarray, Iterable], radius: Union[int, float]) -> None:
        ...

    @overload
    def __init__(self, center: Point, radius: Union[int, float]) -> None:
        ...

    def __init__(self, *args):
        if len(args) == 0:
            self.center = Point()
            self.radius = 0.0
        elif isinstance(args[1], (int, float)):
            if isinstance(args[0], (Point, np.ndarray, Iterable)):
                self.center = Point(args[0])
                self.radius = args[1]
        else:
            raise ValueError("bad argument to constructor")

    def get_center(self) -> Point:
        return copy.deepcopy(self.center)

    def get_radius(self) -> float:
        return self.radius
