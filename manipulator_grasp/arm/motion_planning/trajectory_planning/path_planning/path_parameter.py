from abc import ABC

from manipulator_grasp.arm.interface import Parameter


class PathParameter(Parameter, ABC):
    def get_length(self):
        pass
