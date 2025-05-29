import numpy as np

from arm.interface import ModeEnum
from ..position_parameter import PositionParameter
from ..position_planning_mode_enum import PositionPlanningModeEnum


class ArcCenterPositionParameter(PositionParameter):

    def __init__(self, t0: np.ndarray, t1: np.ndarray, tc: np.ndarray):
        super().__init__()

        self.t0 = t0
        self.t1 = t1
        self.tc = tc

    @classmethod
    def get_mode(cls) -> ModeEnum:
        return PositionPlanningModeEnum.ARC_CENTER

    def get_t0(self) -> np.ndarray:
        return self.t0

    def get_t1(self) -> np.ndarray:
        return self.t1

    def get_tc(self) -> np.ndarray:
        return self.tc

    # TODO: the length of an arc
    def get_length(self):
        super().get_length()
