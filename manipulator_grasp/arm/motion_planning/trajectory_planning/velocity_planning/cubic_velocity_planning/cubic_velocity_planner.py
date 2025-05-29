import numpy as np

from arm.interface import ModeEnum
from ..velocity_planning_mode_enum import VelocityPlanningModeEnum
from ..velocity_parameter import VelocityParameter
from ..velocity_planner_strategy import VelocityPlannerStrategy


class CubicVelocityPlanner(VelocityPlannerStrategy):

    def __init__(self, parameter: VelocityParameter):
        super().__init__(parameter)

        self.x = np.zeros(4)

        self.plan()

    @classmethod
    def mode(cls) -> ModeEnum:
        return VelocityPlanningModeEnum.CUBIC

    def plan(self):
        A = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [1.0, self.parameter.get_tf(), self.parameter.get_tf() ** 2, self.parameter.get_tf() ** 3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 2 * self.parameter.get_tf(), 3.0 * self.parameter.get_tf() ** 2]
        ])

        b = np.array([
            [0.0],
            [1.0],
            [0.0],
            [0.0]
        ])

        x = np.linalg.inv(A) @ b
        self.x = np.squeeze(x)

    def interpolate(self, t: float):
        if t <= 0.0:
            s = 0.0
        elif t > self.parameter.get_tf():
            s = 1.0
        else:
            s = 0.0
            for i in range(self.x.size):
                s += self.x[i] * (t ** i)
        return s
