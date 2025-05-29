import copy
from typing import Iterable

from ..simplex import Point, Line, LineSegment
from ..shape import Circle2D
from .intersect2d import Intersect2D


class Collision2D:
    def __init__(self, obstacles: Iterable[Circle2D]) -> None:
        super().__init__()
        self.obstacles = copy.deepcopy(obstacles)

    def check_point(self, point: Point) -> bool:
        for obstacle in self.obstacles:
            if Intersect2D.check_point_to_circle(point, obstacle):
                return True
        return False

    def check_line(self, line: Line):
        for obstacle in self.obstacles:
            if Intersect2D.check_line_to_circle(line, obstacle):
                return True
        return False

    def check_line_segment(self, line_segment: LineSegment):
        for obstacle in self.obstacles:
            if Intersect2D.check_line_segment_to_circle(line_segment, obstacle):
                return True
        return False
