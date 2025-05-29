import copy
from typing import List

import numpy as np

from .geometry import Geometry
from .simplex import Simplex
from .interface import Support
from .point import Point
from .vector import Vector


class Triangle(Simplex, Support):

    def __init__(self, points: List[Point]) -> None:
        super().__init__()
        self.__points = copy.deepcopy(points)

    @property
    def points(self) -> List[Point]:
        return copy.deepcopy(self.__points)

    def calculate_closest_point_to_origin(self) -> Geometry:
        a = self.points[0]
        b = self.points[1]
        c = self.points[2]

        origin = Point(np.zeros_like(a.get_t()))

        ab = Vector(a, b)
        ac = Vector(a, c)
        ap = Vector(a, origin)

        d1 = np.dot(ab.get_t(), ap.get_t())
        d2 = np.dot(ac.get_t(), ap.get_t())
        if d1 <= 0.0 and d2 <= 0.0:
            return a

        bp = Vector(b, origin)
        d3 = np.dot(ab.get_t(), bp.get_t())
        d4 = np.dot(ac.get_t(), bp.get_t())
        if d3 >= 0.0 and d4 <= d3:
            return b

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / (d1 - d3)
            return a + v * ab

        cp = Vector(c, origin)
        d5 = np.dot(ab.get_t(), cp.get_t())
        d6 = np.dot(ac.get_t(), cp.get_t())
        if d6 >= 0.0 and d5 <= d6:
            return c

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / (d2 - d6)
            return a + w * ac

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and d4 - d3 > 0.0 and d5 - d6 >= 0.0:
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
            return b + w * (c - b)

        denom = 1.0 / (va + vb + vc)
        v = vb * denom
        w = vc * denom
        return a + ab * v + ac * w

    def calculate_barycentric_coordinates(self, geometry: Geometry) -> List[float]:
        v0 = (self.points[1] - self.points[0]).get_t()
        v1 = (self.points[2] - self.points[0]).get_t()
        v2 = (geometry - self.points[0]).get_t()

        if np.array_equal(v0, v2):
            return [0.0, 1.0, 0.0]
        elif np.array_equal(v1, v2):
            return [0.0, 0.0, 1.0]

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        return [u, v, w]
