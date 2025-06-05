from enum import unique
from manipulator_grasp.arm.interface import ModeEnum


@unique
class VelocityPlanningModeEnum(ModeEnum):
    CUBIC = 'cubic'
    QUINTIC = 'quintic'
    T_CURVE = 't_curve'
