import copy
from typing import Union, List, Tuple
import multiprocessing

from arm.geometry import LineSegment, Collision

from .i_check_collision import ICheckCollision


class CheckCollision(ICheckCollision):

    def __init__(self, obstacles: Union[List, Tuple]) -> None:
        self._obstacles = copy.deepcopy(obstacles)

    def check_collision(self, line_segment: LineSegment, pool: multiprocessing.Pool = None):
        for obstacle in self._obstacles:
            if Collision.is_collision(line_segment, obstacle):
                return True
        return False
