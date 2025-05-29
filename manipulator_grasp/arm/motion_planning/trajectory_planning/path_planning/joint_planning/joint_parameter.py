import numpy as np

from manipulator_grasp.arm.interface import ModeEnum
from ..path_planning_mode_enum import PathPlanningModeEnum
from ..path_parameter import PathParameter


class JointParameter(PathParameter):

    def __init__(self, q0: np.ndarray, q1: np.ndarray) -> None:
        super().__init__()

        self.q0 = q0
        self.q1 = q1

    @classmethod
    def get_mode(cls) -> ModeEnum:
        return PathPlanningModeEnum.JOINT

    def get_q0(self) -> np.ndarray:
        return self.q0

    def get_q1(self) -> np.ndarray:
        return self.q1

    def get_t0(self) -> np.ndarray:
        return self.get_q0()

    def get_t1(self) -> np.ndarray:
        return self.get_q1()

    def get_length(self):
        return np.linalg.norm(self.q1 - self.q0)
