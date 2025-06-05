from enum import unique
from manipulator_grasp.arm.interface import ModeEnum


@unique
class AttitudePlannerModeEnum(ModeEnum):
    ONE = 'one'
    TWO = 'two'
    THREE = 'three'
