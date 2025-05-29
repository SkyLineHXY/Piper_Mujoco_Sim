from enum import Enum, unique


@unique
class InterpolatorModeEnum(Enum):
    CUBIC = 'cubic'
    QUINTIC = 'quintic'
    T_CURVE = 't_curve'
