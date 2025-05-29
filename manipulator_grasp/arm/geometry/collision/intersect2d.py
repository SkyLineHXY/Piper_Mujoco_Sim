from ..simplex import Point, Line, LineSegment
from ..shape import Circle2D
from .distance2d import Distance2D


class Intersect2D:

    @staticmethod
    def check_point_to_circle(point: Point, circle: Circle2D) -> bool:
        return Distance2D.point_to_circle(point, circle) <= 0.0

    @staticmethod
    def check_line_to_circle(line: Line, circle: Circle2D):
        return Distance2D.line_to_circle(line, circle) <= 0.0

    @staticmethod
    def check_line_segment_to_circle(line_segment: LineSegment, circle: Circle2D):
        return Distance2D.line_segment_to_circle(line_segment, circle) <= 0.0
