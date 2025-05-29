from arm.interface import ModeEnum
from ..velocity_parameter import VelocityParameter
from ..velocity_planning_mode_enum import VelocityPlanningModeEnum


class CubicVelocityParameter(VelocityParameter):

    def __init__(self, tf: float) -> None:
        super().__init__()

        self.tf = tf

    @classmethod
    def get_mode(cls) -> ModeEnum:
        return VelocityPlanningModeEnum.CUBIC

    def get_tf(self) -> float:
        return self.tf
