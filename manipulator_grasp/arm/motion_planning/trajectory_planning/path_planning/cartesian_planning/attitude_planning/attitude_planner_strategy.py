from abc import ABC, abstractmethod

from spatialmath import SO3

from arm.interface import Strategy

from .attitude_parameter import AttitudeParameter


class AttitudePlannerStrategy(Strategy, ABC):

    def __init__(self, parameter: AttitudeParameter):
        super().__init__(parameter)

    @abstractmethod
    def plan(self) -> None:
        pass

    @abstractmethod
    def interpolate(self, s) -> SO3:
        pass
