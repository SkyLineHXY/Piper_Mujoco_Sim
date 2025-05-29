from abc import ABC, abstractmethod

import numpy as np
from spatialmath import SO3

from .mode_enum import ModeEnum


class Parameter(ABC):

    @classmethod
    @abstractmethod
    def get_mode(cls) -> ModeEnum:
        pass

    def get_tf(self) -> float:
        pass

    def get_t0(self) -> np.ndarray:
        pass

    def get_t1(self) -> np.ndarray:
        pass

    def get_tc(self) -> np.ndarray:
        pass

    def get_R0(self) -> SO3:
        pass

    def get_R1(self) -> SO3:
        pass

    def get_vec_n(self) -> np.ndarray:
        pass

    def get_q0(self) -> np.ndarray:
        pass

    def get_q1(self) -> np.ndarray:
        pass
