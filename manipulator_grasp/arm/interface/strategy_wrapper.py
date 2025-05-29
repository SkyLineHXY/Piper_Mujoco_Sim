from abc import ABC, abstractmethod
from typing import Type

from .parameter import Parameter
from .strategy import Strategy
from .factory import Factory


class StrategyWrapper(ABC):
    def __init__(self, parameter: Parameter) -> None:
        self.strategy: Strategy = self.get_factory().strategy.get(parameter.get_mode())(parameter)

    @staticmethod
    def get_factory() -> Type[Factory]:
        return Factory
