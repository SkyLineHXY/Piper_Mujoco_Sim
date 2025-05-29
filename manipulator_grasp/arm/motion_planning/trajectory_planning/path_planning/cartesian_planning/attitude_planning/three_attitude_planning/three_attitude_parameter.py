import numpy as np
from spatialmath import SO3

from ..attitude_parameter import AttitudeParameter
from ..attitude_planner_mode_enum import AttitudePlannerModeEnum


class ThreeAttitudeParameter(AttitudeParameter):

    def __init__(self, R0: SO3, R1: SO3, vec_n: np.ndarray) -> None:
        self.R0 = R0
        self.R1 = R1
        self.vec_n = vec_n / np.linalg.norm(vec_n)

    @classmethod
    def get_mode(cls) -> AttitudePlannerModeEnum:
        return AttitudePlannerModeEnum.THREE

    def get_R0(self) -> SO3:
        return self.R0

    def get_R1(self) -> SO3:
        return self.R1

    def get_vec_n(self) -> np.ndarray:
        return self.vec_n
