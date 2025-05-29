from enum import unique
from arm.interface import ModeEnum


@unique
class PositionPlanningModeEnum(ModeEnum):
    LINE = 'line'
    ARC_CENTER = 'arc_center'
    ARC_POINT = 'arc_point'
