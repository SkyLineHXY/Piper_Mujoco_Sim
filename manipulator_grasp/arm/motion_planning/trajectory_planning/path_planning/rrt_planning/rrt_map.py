import copy
from typing import Union, List, Tuple


class RRTMap:

    def __init__(self, area: Union[List, Tuple],
                 obstacles: Union[List, Tuple]) -> None:
        super().__init__()
        self.__area = copy.deepcopy(area)
        self.__obstacles = copy.deepcopy(obstacles)

    @property
    def area(self) -> Union[List, Tuple]:
        return copy.deepcopy(self.__area)

    @property
    def obstacles(self) -> Union[List, Tuple]:
        return copy.deepcopy(self.__obstacles)
