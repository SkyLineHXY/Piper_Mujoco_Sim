import numpy as np

from arm.interface import ModeEnum
from ..velocity_planning_mode_enum import VelocityPlanningModeEnum
from ..velocity_parameter import VelocityParameter
from ..velocity_planner_strategy import VelocityPlannerStrategy


class QuinticVelocityPlanner(VelocityPlannerStrategy):

    def __init__(self, parameter: VelocityParameter):
        super().__init__(parameter)

        self.x = np.zeros(6)

        self.plan()

    @classmethod
    def mode(cls) -> ModeEnum:
        return VelocityPlanningModeEnum.QUINTIC

    def plan(self):
        A = np.zeros((6, 6))
        A[0, 0] = 1.0
        A[2, 1] = 1.0
        A[4, 2] = 2.0
        for i in range(6):
            A[1, i] = (self.parameter.get_tf() ** i)
            A[3, i] = i * (self.parameter.get_tf() ** (i - 1))
            A[5, i] = (i - 1) * i * (self.parameter.get_tf() ** (i - 2))

        b = np.zeros((6, 1))
        b[1] = 1.0

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
