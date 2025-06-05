from abc import ABC

from manipulator_grasp.arm.interface import Strategy, Parameter


class VelocityPlannerStrategy(Strategy, ABC):

    def __init__(self, parameter: Parameter):
        super().__init__(parameter)
