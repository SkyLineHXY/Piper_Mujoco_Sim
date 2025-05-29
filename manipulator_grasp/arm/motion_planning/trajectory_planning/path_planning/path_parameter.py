from abc import ABC

from arm.interface import Parameter


class PathParameter(Parameter, ABC):
    def get_length(self):
        pass
