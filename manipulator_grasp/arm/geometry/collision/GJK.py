import copy
from typing import Tuple

import numpy as np

from manipulator_grasp.arm.constanst import MathConst
from ..simplex import Point, SimplexFactoryPool, SimplexParameter, UnitVector, Support


class GJK:

    @staticmethod
    def calculate_distance(shape0: Support, shape1: Support) -> float:

        return GJK.calculate_distance_and_points(shape0, shape1)[0]

    @staticmethod
    def calculate_distance_and_points(shape0: Support, shape1: Support) -> Tuple[float, Tuple]:
        vec = UnitVector(np.array([1, 0, 0]))
        point0 = shape0.calculate_support_point(vec)
        point1 = shape1.calculate_support_point(-vec)
        point = point0 - point1

        origin = Point([0, 0, 0])

        closest_point = point
        comparator = lambda x: np.linalg.norm(x.get_t())
        points = [point]
        points0 = [point0]
        points1 = [point1]

        coordinates = [1]

        finish = False

        while closest_point != origin:
            closest = copy.deepcopy(closest_point)
            vec = -UnitVector(closest_point)
            point0 = shape0.calculate_support_point(vec)
            point1 = shape1.calculate_support_point(-vec)
            point = point0 - point1

            for point_i in points:
                if np.linalg.norm((point - point_i).get_t()) < MathConst.ERROR:
                    finish = True
                    break
            if finish:
                break

            points.append(point)
            points0.append(point0)
            points1.append(point1)

            if len(points) == 5:
                coordinate_min = min(coordinates)
                coordinate_min_index = coordinates.index(coordinate_min)
                points.pop(coordinate_min_index)
                points0.pop(coordinate_min_index)
                points1.pop(coordinate_min_index)
                break

            simplex_parameter = SimplexParameter(points)
            simplex = SimplexFactoryPool.create_product(simplex_parameter)
            closest_point = simplex.calculate_closest_point_to_origin()
            coordinates = simplex.calculate_barycentric_coordinates(closest_point)

            if np.linalg.norm((closest - closest_point).get_t()) < MathConst.EPS:
                break

            j = 0
            for i, coordinate in enumerate(coordinates):
                if abs(coordinate) < MathConst.EPS:
                    points.pop(i - j)
                    points0.pop(i - j)
                    points1.pop(i - j)
                    j = j + 1

        simplex_parameter = SimplexParameter(points)
        simplex = SimplexFactoryPool.create_product(simplex_parameter)
        coordinates = simplex.calculate_barycentric_coordinates(closest_point)

        point0 = Point(np.zeros_like(point0.get_t()))
        point1 = Point(np.zeros_like(point1.get_t()))
        for i, coordinate_i in enumerate(coordinates):
            point0 += coordinate_i * points0[i]
            point1 += coordinate_i * points1[i]

        return np.linalg.norm(closest_point.get_t()), (point0, point1)

    @staticmethod
    def is_intersecting(shape0: Support, shape1: Support):
        origin = Point([0, 0, 0])
        unit_vector = UnitVector(np.array([1, 0, 0]))

        point = shape0.calculate_support_point(unit_vector) - shape1.calculate_support_point(-unit_vector)
        points = [point]
        closest_point = point

        coordinates = [1]

        while closest_point != origin:
            unit_vector = -UnitVector(closest_point)
            point = shape0.calculate_support_point(unit_vector) - shape1.calculate_support_point(-unit_vector)
            if np.dot(point.get_t(), unit_vector.get_t()) < 0:
                return False
            points.append(point)

            if len(points) == 5:
                coordinate_min = min(coordinates)
                coordinate_min_index = coordinates.index(coordinate_min)
                points.pop(coordinate_min_index)
                break

            simplex_parameter = SimplexParameter(points)
            simplex = SimplexFactoryPool.create_product(simplex_parameter)
            closest_point = simplex.calculate_closest_point_to_origin()
            coordinates = simplex.calculate_barycentric_coordinates(closest_point)

            j = 0
            for i, coordinate in enumerate(coordinates):
                if abs(coordinate) < MathConst.EPS:
                    points.pop(i - j)
                    j = j + 1

        return True
