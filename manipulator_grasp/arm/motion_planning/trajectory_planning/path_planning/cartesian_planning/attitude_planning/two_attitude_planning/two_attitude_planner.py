import numpy as np
from spatialmath import SO3

from ..attitude_planner_mode_enum import AttitudePlannerModeEnum
from ..attitude_planner_strategy import AttitudePlannerStrategy

from .two_attitude_parameter import TwoAttitudeParameter


class TwoAttitudePlanner(AttitudePlannerStrategy):

    def __init__(self, parameter: TwoAttitudeParameter):
        super().__init__(parameter)

        self.theta_d = 0.0
        self.axis_d = np.zeros(3)
        self.theta_s = 0.0
        self.axis_s = np.zeros(3)

        self.plan()

    @classmethod
    def mode(cls) -> AttitudePlannerModeEnum:
        return AttitudePlannerModeEnum.TWO

    def plan(self) -> None:
        self.theta_d = np.arccos(np.dot(self.parameter.get_R0().a, self.parameter.get_R1().a))
        self.axis_d = np.cross(self.parameter.get_R0().a, self.parameter.get_R1().a)
        self.axis_d = self.axis_d / np.linalg.norm(self.axis_d)

        Rd = SO3.AngVec(self.theta_d, self.axis_d)
        Rs = (Rd * self.parameter.get_R0()).inv() * self.parameter.get_R1()

        self.theta_s = np.arctan2(Rs.n[1], Rs.n[0])
        self.axis_s = np.array([0, 0, 1])

    def interpolate(self, s) -> SO3:
        return SO3.AngVec(s * self.theta_d, self.axis_d) * self.parameter.get_R0() \
               * SO3.AngVec(s * self.theta_s, self.axis_s)
