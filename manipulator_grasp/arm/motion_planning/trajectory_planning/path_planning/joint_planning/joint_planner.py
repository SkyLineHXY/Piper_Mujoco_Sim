import numpy as np

from arm.interface import ModeEnum
from ..path_planner_strategy import PathPlannerStrategy

from ..path_planning_mode_enum import PathPlanningModeEnum

from .joint_parameter import JointParameter


class JointPlanner(PathPlannerStrategy):

    def __init__(self, parameter: JointParameter):
        super().__init__(parameter)

        self.plan()

    @classmethod
    def mode(cls) -> ModeEnum:
        return PathPlanningModeEnum.JOINT

    def plan(self):
        pass

    def interpolate(self, s: float):
        return self.parameter.get_q0() + s * (self.parameter.get_q1() - self.parameter.get_q0())

    def interpolate_derivative(self, s: float):
        return self.parameter.get_q1() - self.parameter.get_q0()

    def interpolate_second_derivative(self, s: float):
        return np.zeros_like(self.parameter.get_q0())
