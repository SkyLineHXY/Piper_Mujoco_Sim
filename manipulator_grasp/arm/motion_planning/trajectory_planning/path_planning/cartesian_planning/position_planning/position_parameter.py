from abc import ABC

from arm.interface import Parameter


class PositionParameter(Parameter, ABC):
    def get_length(self):
        pass
