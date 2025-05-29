from abc import ABC, abstractmethod

import numpy as np

from arm.interface import Strategy

from .position_parameter import PositionParameter


class PositionPlannerStrategy(Strategy, ABC):

    def __init__(self, parameter: PositionParameter):
        super().__init__(parameter)

    @abstractmethod
    def plan(self) -> None:
        pass

    @abstractmethod
    def interpolate(self, s) -> np.ndarray:
        pass
