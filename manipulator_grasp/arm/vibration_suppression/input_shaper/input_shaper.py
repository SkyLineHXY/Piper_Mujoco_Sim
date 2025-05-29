import abc

import numpy as np


class InputShaper(abc.ABC):

    def __init__(self, omega_d: float, zeta: float, ts: float) -> None:
        super().__init__()

        self._omega_d = omega_d
        self._zeta = zeta
        self._ts = ts

        self._delay_t = np.pi / self._omega_d
        self._delay_count = int(self._delay_t / self._ts)

        self._K = 0
        self._as = []

    @abc.abstractmethod
    def shape(self, traj: np.ndarray) -> np.ndarray:
        pass

    @property
    def delay_count(self) -> int:
        return 0

    @property
    def omega_d(self) -> float:
        return self._omega_d

    @property
    def zeta(self) -> float:
        return self._zeta

    @property
    def ts(self) -> float:
        return self._ts
