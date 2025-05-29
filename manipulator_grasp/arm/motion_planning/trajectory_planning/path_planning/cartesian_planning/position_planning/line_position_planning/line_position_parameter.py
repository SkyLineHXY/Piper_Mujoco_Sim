import numpy as np

from arm.interface import ModeEnum
from ..position_parameter import PositionParameter
from ..position_planning_mode_enum import PositionPlanningModeEnum


class LinePositionParameter(PositionParameter):

    def __init__(self, t0: np.ndarray, t1: np.ndarray):
        self.t0 = t0
        self.t1 = t1

    @classmethod
    def get_mode(cls) -> ModeEnum:
        return PositionPlanningModeEnum.LINE

    def get_t0(self) -> np.ndarray:
        return self.t0

    def get_t1(self) -> np.ndarray:
        return self.t1

    def get_length(self):
        return np.linalg.norm(self.t1 - self.t0)
