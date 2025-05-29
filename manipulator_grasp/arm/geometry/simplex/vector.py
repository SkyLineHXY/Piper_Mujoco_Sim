from typing import overload, Union, Iterable

import numpy as np

from .geometry import Geometry
from .point import Point


class Vector(Geometry):

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
        super().__init__()
        if t0 is None:
            self.t = np.zeros(1)
        elif isinstance(t0, (np.ndarray, Iterable, int, float)):
            if t1 is None:
                self.t = np.array(t0, dtype=np.float64)
            elif isinstance(t1, (np.ndarray, Iterable, int, float)):
                self.t = np.array(t1, dtype=np.float64) - np.array(t0, dtype=np.float64)
        elif isinstance(t0, Point):
            if t1 is None:
                self.t = t0.get_t()
            elif isinstance(t1, Point):
                self.t = t1.get_t() - t0.get_t()

        else:
            raise ValueError("bad argument to constructor")

    def __add__(self, other: Geometry):
        return Vector(np.zeros_like(self.t), self.t + other.get_t())

    def __sub__(self, other: Geometry):
        return Vector(np.zeros_like(self.t), self.t - other.get_t())

    def __mul__(self, other):
        return Vector(np.zeros_like(self.t), other * self.t)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Vector(np.zeros_like(self.t), self.t / other)

    def __neg__(self):
        return self.__class__(-self.t)

    def norm(self) -> float:
        return np.linalg.norm(self.t)


if __name__ == '__main__':
    t0_0 = np.array([1, 2, 3])
    t1_0 = np.array([2, 3, 4])
    vector0 = Vector(t0_0, t1_0)

    t0_1 = np.array([3, 4, 5])
    t1_1 = np.array([7, 6, 5])
    vector1 = Vector(t0_1, t1_1)

    vector2 = vector0 + vector1
    print(vector2.get_t())

    vector3 = Vector()
    print("vector3: ", vector3.get_t())

    vector4 = Vector((4, 4, 4))
    print("vector4: ", vector4.get_t())

    vector5 = Vector((4, 4, 4), [5, 5, 5])
    print("vector5: ", vector5.get_t())
