from abc import ABC

from manipulator_grasp.arm.interface import Strategy

from .path_parameter import PathParameter


class PathPlannerStrategy(Strategy, ABC):

    def __init__(self, parameter: PathParameter):
        super().__init__(parameter)
