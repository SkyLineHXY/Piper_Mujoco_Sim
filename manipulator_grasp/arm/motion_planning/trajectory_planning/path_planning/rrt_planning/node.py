import copy

from typing import overload, Union, Iterable

import numpy as np

from arm.geometry import Point


class Node:

    @overload
    def __init__(self, point: Union[np.ndarray, Iterable], cost: float = 0.0, parent: int = -1):
        ...

    @overload
    def __init__(self, point: Point, cost: float = 0.0, parent: int = -1):
        ...

    def __init__(self, point=None, cost=0.0, parent=-1):
        if point is None:
            self.point = Point()
        elif isinstance(point, (Point, np.ndarray, Iterable)):
            self.point = Point(point)
        # self.point = point
        self.cost: float = cost
        self.parent: int = parent

    def get_tx(self) -> float:
        return self.point.get_tx()

    def get_ty(self) -> float:
        return self.point.get_ty()

    def get_tz(self) -> float:
        return self.point.get_tz()

    def get_t(self) -> np.ndarray:
        return self.point.get_t()

    def get_t_3d(self) -> np.ndarray:
        return self.point.get_t_3d()

    def set_tx(self, num: float):
        self.point.set_tx(num)

    def set_ty(self, num: float):
        self.point.set_ty(num)

    def set_tz(self, num: float):
        self.point.set_tz(num)

    def get_point(self) -> Point:
        return copy.deepcopy(self.point)

    def get_cost(self) -> float:
        return self.cost

    def get_parent(self) -> int:
        return self.parent

    def set_cost(self, cost: float):
        self.cost = cost

    def set_parent(self, parent: int):
        self.parent = parent
