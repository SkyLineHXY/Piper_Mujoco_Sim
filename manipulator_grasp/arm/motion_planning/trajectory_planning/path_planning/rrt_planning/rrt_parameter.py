import copy
from typing import Union, Iterable

import numpy as np

from .i_check_collision import ICheckCollision
from .rrt_map import RRTMap
from .check_collision import CheckCollision


class RRTParameter:

    def __init__(self, start: Union[np.ndarray, Iterable], goal: Union[np.ndarray, Iterable],
                 expand_dis: float = 1.0, goal_sample_rate: float = 10.0, max_iter: int = 100,
                 radius: float = 10.0, animation: bool = False) -> None:
        super().__init__()
        self.__start = np.array(start)
        self.__goal = np.array(goal)
        self.__expand_dis = expand_dis
        self.__goal_sample_rate = goal_sample_rate
        self.__max_iter = max_iter
        self.__radius = radius
        self.__animation = animation

    @property
    def start(self):
        return copy.deepcopy(self.__start)

    @property
    def goal(self):
        return copy.deepcopy(self.__goal)

    @property
    def expand_dis(self):
        return self.__expand_dis

    @property
    def goal_sample_rate(self):
        return self.__goal_sample_rate

    @property
    def max_iter(self):
        return self.__max_iter

    @property
    def radius(self):
        return self.__radius

    @property
    def animation(self):
        return self.__animation

    def create_check_collision(self, rrt_map: RRTMap) -> ICheckCollision:
        return CheckCollision(rrt_map.obstacles)
