import numpy as np
from spatialmath import SO3

from ..attitude_planner_mode_enum import AttitudePlannerModeEnum
from ..attitude_planner_strategy import AttitudePlannerStrategy

from .three_attitude_parameter import ThreeAttitudeParameter


class ThreeAttitudePlanner(AttitudePlannerStrategy):

    def __init__(self, parameter: ThreeAttitudeParameter):
        super().__init__(parameter)

        self.theta_a = 0.0
        self.axis_a = np.zeros(3)
        self.theta_e = 0.0
        self.axis_e = np.zeros(3)
        self.theta_s = 0.0
        self.axis_s = np.zeros(3)

        self.plan()

    @classmethod
    def mode(cls) -> AttitudePlannerModeEnum:
        return AttitudePlannerModeEnum.THREE

    def plan(self) -> None:
        vz0 = self.parameter.get_R0().a
        vz1 = self.parameter.get_R1().a

        vzp0 = vz0 - np.dot(vz0, self.parameter.get_vec_n()) * self.parameter.get_vec_n()
        vzp0 = vzp0 / np.linalg.norm(vzp0)

        vzp1 = vz1 - np.dot(vz1, self.parameter.get_vec_n()) * self.parameter.get_vec_n()
        vzp1 = vzp1 / np.linalg.norm(vzp1)

        self.theta_a = np.arccos(np.dot(vzp0, vzp1))
        self.axis_a = np.cross(vzp0, vzp1)
        self.axis_a = self.axis_a / np.linalg.norm(self.axis_a)

        Ra = SO3.AngleAxis(self.theta_a, self.axis_a)

        Ra0 = Ra * self.parameter.get_R0()
        vz_a0 = Ra0.a
        self.theta_e = np.arccos(np.dot(vz_a0, vz1))

        axis_we = np.cross(vz_a0, vz1)
        axis_we = axis_we / np.linalg.norm(axis_we)
        self.axis_e = Ra0.inv() * axis_we

        Re = SO3.AngleAxis(self.theta_e, self.axis_e)

        R_a0e = Ra * self.parameter.get_R0() * Re
        Rs = R_a0e.inv() * self.parameter.get_R1()

        self.theta_s = np.arctan2(Rs.n[1], Rs.n[0])
        self.axis_s = np.array([0, 0, 1])

    def interpolate(self, s) -> SO3:
        Ri = SO3.AngleAxis(s * self.theta_a, self.axis_a) * self.parameter.get_R0() \
             * SO3.AngleAxis(s * self.theta_e, self.axis_e) * SO3.AngleAxis(s * self.theta_s, self.axis_s)

        return Ri
