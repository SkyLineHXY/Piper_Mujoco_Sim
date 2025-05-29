from abc import ABC, abstractmethod
from typing import Type

from arm.utils import ClassUtils
from . import Parameter

from .register import Register
from .factory import Factory


class Strategy(Register, ABC):

    def __init__(self, parameter: Parameter):
        self.parameter = parameter

    @staticmethod
    def factory() -> Type[Factory]:
        return Factory

    @classmethod
    def register(cls) -> None:
        cls.factory().register(cls.mode(), cls)

    @classmethod
    def factory_register(cls):
        leaf_subclasses = ClassUtils.get_leaf_subclasses(cls)
        for subclass in leaf_subclasses:
            subclass.register()

    def interpolate(self, s: float):
        pass
