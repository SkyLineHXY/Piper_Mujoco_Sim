from typing import List
from abc import ABC, abstractmethod
import numpy as np

from manipulator_grasp.arm.geometry.simplex.geometry import Geometry


class Support(ABC):
    @property
    @abstractmethod
    def points(self) -> List[Geometry]:
        pass

    def calculate_support_point(self, d: Geometry) -> Geometry:
        dot = -np.inf
        support_point = self.points[0]
        for point in self.points:
            dot_new = np.dot(point.get_t(), d.get_t())
            if dot_new > dot:
                dot = dot_new
                support_point = point

        return support_point
