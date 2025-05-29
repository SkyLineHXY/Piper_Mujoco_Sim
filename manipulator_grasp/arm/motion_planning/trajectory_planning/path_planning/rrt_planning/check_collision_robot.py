import copy
import math
import multiprocessing
from typing import Union, List, Tuple
from functools import partial

from arm.geometry import LineSegment, Collision
from arm.robot import Robot

from .check_collision import CheckCollision


class CheckCollisionRobot(CheckCollision):
    count = 10

    def __init__(self, obstacles: Union[List, Tuple], expand_dis: float, robot: Robot) -> None:
        super().__init__(obstacles)
        self.__expand_dis = expand_dis
        self.__robot = copy.deepcopy(robot)

    def check_collision(self, line_segment: LineSegment, pool: multiprocessing.Pool = None):
        count = math.ceil(line_segment.get_length() * CheckCollisionRobot.count / self.__expand_dis)
        if count == 0:
            count += 1

        if pool:
            func = partial(self._check_collision_robot_joint, line_segment, count)
            results = pool.imap_unordered(func, range(count + 1))
            for result in results:
                if result:
                    return True
            return False

        for i in range(count + 1):
            if self._check_collision_robot_joint(line_segment, count, i):
                return True
        return False

    def _check_collision_robot_joint(self, line_segment: LineSegment, count: int, num: int):
        q0 = line_segment.get_point0().get_t()
        q1 = line_segment.get_point1().get_t()

        q = q0 + (q1 - q0) / count * num
        self.__robot.set_joint(q)
        geometries = self.__robot.get_geometries()
        for obstacles in self._obstacles:
            for geometry in geometries:
                if Collision.is_collision(obstacles, geometry):
                    return True
        return False
