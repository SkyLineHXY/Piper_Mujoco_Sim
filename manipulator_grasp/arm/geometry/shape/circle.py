from typing import List

import numpy as np
from spatialmath import SE3

from manipulator_grasp.arm.constanst import MathConst
from ..simplex import Point, UnitVector, Support
from .geometry3d import Geometry3D


class Circle(Geometry3D, Support):

    def __init__(self, base: SE3, radius: float) -> None:
        super().__init__(base)
        self.radius = radius

    @property
    def points(self) -> List[Point]:
        return []

    @property
    def normal_vector(self) -> UnitVector:
        return UnitVector(self.base.a)

    def calculate_support_point(self, d: UnitVector) -> Point:
        vector = (d - np.dot(self.normal_vector.get_t(), d.get_t()) * self.normal_vector)
        if vector.norm() < MathConst.EPS:
            return Point(self.base.t + self.radius * np.array([1.0, 0.0, 0.0]))

        projection_vector = UnitVector(vector.get_t())

        return Point(self.base.t + self.radius * projection_vector.get_t())
