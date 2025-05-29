import copy
from typing import List
import numpy as np

from .geometry import Geometry
from .simplex import Simplex
from .interface import Support
from .point import Point
from .triangle import Triangle


class Tetrahedron(Simplex, Support):
    def __init__(self, points: List[Point]) -> None:
        super().__init__()
        self.__points = copy.deepcopy(points)

    @property
    def points(self) -> List[Point]:
        return copy.deepcopy(self.__points)

    def calculate_closest_point_to_origin(self) -> Geometry:
        point = Point(np.zeros_like(self.__points[0].get_t()))

        closest_point = point
        best_sq_dist = np.inf

        for i in range(4):

            if self.point_outside_of_plane(point, i):
                points = copy.deepcopy(self.__points)
                points.pop(i)

                triangle = Triangle(points)
                q = triangle.calculate_closest_point_to_origin()
                sq_dist = np.dot((q - point).get_t(), (q - point).get_t())
                if sq_dist < best_sq_dist:
                    best_sq_dist = sq_dist
                    closest_point = q

        return closest_point

    def point_outside_of_plane(self, point: Point, index: int) -> bool:
        points = copy.deepcopy(self.__points)
        d = points.pop(index)
        other_points = points

        vec_n = np.cross((other_points[0] - other_points[1]).get_t(), (other_points[0] - other_points[2]).get_t())
        sign_p = np.dot((point - other_points[0]).get_t(), vec_n)
        sign_d = np.dot((d - other_points[0]).get_t(), vec_n)

        return sign_p * sign_d <= 0.0

    def calculate_barycentric_coordinates(self, geometry: Geometry) -> List[float]:
        v0 = (self.points[1] - self.points[0]).get_t()
        v1 = (self.points[2] - self.points[0]).get_t()
        v2 = (self.points[3] - self.points[0]).get_t()
        v3 = (geometry - self.points[0]).get_t()

        A = np.vstack((v0, v1, v2)).T
        b = np.reshape(v3, (v3.size, -1))

        x = np.squeeze(np.linalg.pinv(A) @ b)

        return [1.0 - np.sum(x), *x]
