import numpy as np

from .input_shaper import InputShaper


class ZVShaper(InputShaper):

    def __init__(self, omega_d: float, zeta: float, ts: float) -> None:
        super().__init__(omega_d, zeta, ts)

        self._K = np.exp((np.pi * self._zeta) / (np.sqrt(1 - np.power(self._zeta, 2))))
        self._as.append(self._K / (self._K + 1.0))
        self._as.append(1 - self._as[0])

    def shape(self, traj: np.ndarray) -> np.ndarray:
        shaped_traj = np.array([])
        if len(traj.shape) == 2:
            shaped_traj = np.zeros((traj.shape[0] + self._delay_count, traj.shape[1]))
            for i in range(shaped_traj.shape[0]):
                if i < self._delay_count:
                    shaped_traj[i, :] = self._as[0] * (traj[i, :] - traj[0, :]) + traj[0, :]
                elif i >= traj.shape[0]:
                    shaped_traj[i, :] = self._as[0] * traj[-1, :] + self._as[1] * traj[i - self._delay_count, :]
                else:
                    shaped_traj[i, :] = self._as[0] * traj[i, :] + self._as[1] * traj[i - self._delay_count, :]
        elif len(traj.shape) == 1:
            shaped_traj = np.zeros((traj.shape[0] + self._delay_count))
            for i in range(shaped_traj.shape[0]):
                if i < self._delay_count:
                    shaped_traj[i] = self._as[0] * (traj[i] - traj[0]) + traj[0]
                elif i >= traj.shape[0]:
                    shaped_traj[i] = self._as[0] * traj[-1] + self._as[1] * traj[i - self._delay_count]
                else:
                    shaped_traj[i] = self._as[0] * traj[i] + self._as[1] * traj[i - self._delay_count]
        return shaped_traj

    @property
    def delay_count(self) -> int:
        return self._delay_count
