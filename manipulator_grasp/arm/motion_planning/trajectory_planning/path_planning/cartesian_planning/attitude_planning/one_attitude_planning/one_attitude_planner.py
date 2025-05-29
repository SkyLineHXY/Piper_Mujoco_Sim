import numpy as np
from spatialmath import SO3

from ..attitude_planner_mode_enum import AttitudePlannerModeEnum
from ..attitude_planner_strategy import AttitudePlannerStrategy

from .one_attitude_parameter import OneAttitudeParameter


class OneAttitudePlanner(AttitudePlannerStrategy):

    def __init__(self, parameter: OneAttitudeParameter):
        super().__init__(parameter)

        self.theta = 0.0
        self.axis = np.array(3)

        self.plan()

    @classmethod
    def mode(cls) -> AttitudePlannerModeEnum:
        return AttitudePlannerModeEnum.ONE

    def plan(self) -> None:
        R = self.parameter.get_R0().inv() * self.parameter.get_R1()
        trace_R = np.trace(R)
        if trace_R >= 3:
            self.theta = 0
            self.axis = np.array([1, 0, 0])
        else:
            self.theta, self.axis = (self.parameter.get_R0().inv() * self.parameter.get_R1()).angvec()

    def interpolate(self, s) -> SO3:
        return self.parameter.get_R0() * SO3.AngleAxis(s * self.theta, self.axis)
