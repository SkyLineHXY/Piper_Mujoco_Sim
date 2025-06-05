from enum import unique
from manipulator_grasp.arm.interface import ModeEnum


@unique
class PathPlanningModeEnum(ModeEnum):
    JOINT = 'joint'
    CARTESIAN = 'cartesian'
