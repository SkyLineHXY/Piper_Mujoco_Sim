from abc import ABC

from manipulator_grasp.arm.interface import Parameter


class PositionParameter(Parameter, ABC):
    def get_length(self):
        pass
