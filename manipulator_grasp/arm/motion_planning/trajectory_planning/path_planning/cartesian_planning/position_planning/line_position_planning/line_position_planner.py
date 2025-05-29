import numpy as np

from arm.interface import ModeEnum
from ..position_planning_mode_enum import PositionPlanningModeEnum
from ..position_planner_strategy import PositionPlannerStrategy

from .line_position_parameter import LinePositionParameter


class LinePositionPlanner(PositionPlannerStrategy):

    def __init__(self, parameter: LinePositionParameter):
        super().__init__(parameter)

        self.plan()

    @classmethod
    def mode(cls) -> ModeEnum:
        return PositionPlanningModeEnum.LINE

    def plan(self) -> None:
        pass

    def interpolate(self, s) -> np.ndarray:
        return self.parameter.get_t0() + s * (self.parameter.get_t1() - self.parameter.get_t0())
