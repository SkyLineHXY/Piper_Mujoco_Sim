import numpy as np

from ..simplex import Point, Line, LineSegment
from ..shape import Circle2D


class Distance2D:
    @staticmethod
    def point_to_point(point0: Point, point1: Point) -> float:
        return np.linalg.norm((point1 - point0).get_t())

    @staticmethod
    def point_to_line_segment(point: Point, line_segment: LineSegment) -> float:
        foot_point, t = Distance2D.__calculate_foot_point(point, line_segment)

        t = max(0, min(1, t))

        projection = line_segment.get_point0().get_t() + t * (
                line_segment.get_point1().get_t() - line_segment.get_point0().get_t())
        return np.linalg.norm(point.get_t() - projection)

    @staticmethod
    def point_to_line(point: Point, line: Line) -> float:
        foot_point, t = Distance2D.__calculate_foot_point(point, line)
        return np.linalg.norm(point.get_t() - foot_point)

    @staticmethod
    def point_to_circle(point: Point, circle: Circle2D) -> float:
        return Distance2D.point_to_point(point, circle.get_center()) - circle.get_radius()

    @staticmethod
    def line_to_circle(line: Line, circle: Circle2D) -> float:
        return Distance2D.point_to_line(circle.get_center(), line) - circle.get_radius()

    @staticmethod
    def line_segment_to_circle(line_segment: LineSegment, circle: Circle2D) -> float:
        return Distance2D.point_to_line_segment(circle.get_center(), line_segment) - circle.get_radius()

    @staticmethod
    def __calculate_foot_point(point: Point, line: Line) -> tuple:
        v = line.get_point0().get_t()
        w = line.get_point1().get_t()
        p = point.get_t()

        if np.array_equal(v, w):
            return np.linalg.norm(p - v), 0
        l2 = (w - v).dot(w - v)

        t = (p - v).dot(w - v) / l2
        foot_point = v + t * (w - v)
        return foot_point, t
