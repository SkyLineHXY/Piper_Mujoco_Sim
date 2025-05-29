from spatialmath import SO3

from ..attitude_parameter import AttitudeParameter
from ..attitude_planner_mode_enum import AttitudePlannerModeEnum


class TwoAttitudeParameter(AttitudeParameter):

    def __init__(self, R0: SO3, R1: SO3):
        self.R0 = R0
        self.R1 = R1

    @classmethod
    def get_mode(cls) -> AttitudePlannerModeEnum:
        return AttitudePlannerModeEnum.TWO

    def get_R0(self) -> SO3:
        return self.R0

    def get_R1(self) -> SO3:
        return self.R1
