import copy
from typing import List

from manipulator_grasp.arm.geometry.simplex.point import Point


class SimplexParameter:
    def __init__(self, simplexes: List[Point]) -> None:
        super().__init__()
        self.__simplexes = copy.deepcopy(simplexes)

    @property
    def key(self) -> str:
        return str(len(self.__simplexes))

    def parameter(self) -> List[Point]:
        return copy.deepcopy(self.__simplexes)
