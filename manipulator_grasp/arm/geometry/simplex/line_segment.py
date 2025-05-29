import copy
from typing import List

import numpy as np

from manipulator_grasp.arm.constanst import MathConst

from .geometry import Geometry
from .simplex import Simplex
from .interface import Support
from .point import Point
from .line import Line


class LineSegment(Line, Simplex, Support):
    def get_length(self) -> float:
        return self.length

    @property
    def points(self) -> List[Point]:
        return copy.deepcopy([self.point0, self.point1])

    def calculate_closest_point_to_origin(self) -> Geometry:
        v = self.point0.get_t()
        w = self.point1.get_t()
        p = np.zeros_like(v)

        if np.array_equal(v, w):
            return Point(v)

        t = (p - v).dot(w - v) / ((w - v).dot(w - v))
        t = max(0.0, min(1.0, t))
        return Point(v + t * (w - v))

    def calculate_barycentric_coordinates(self, geometry: Geometry) -> List[float]:
        v0 = (self.points[1] - self.points[0]).get_t()
        v1 = (geometry - self.points[0]).get_t()

        d00 = np.dot(v0, v0)
        if np.abs(d00) < MathConst.EPS:
            return [1.0, 0.0]

        d01 = np.dot(v0, v1)

        v = d01 / d00
        u = 1.0 - v

        return [u, v]

    def plot(self, ax, c=None):
        ax.plot([self.point0.get_tx(), self.point1.get_tx()],
                [self.point0.get_ty(), self.point1.get_ty()],
                [self.point0.get_tz(), self.point1.get_tz()],
                c='g' if c is None else c)
