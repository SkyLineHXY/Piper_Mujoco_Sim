import numpy as np
import scipy as sp

from .input_shaper import InputShaper


class NoDelayShaper(InputShaper):
    def __init__(self, shaper: InputShaper) -> None:
        super().__init__(shaper.omega_d, shaper.zeta, shaper.ts)
        self._shaper = shaper

    def shape(self, traj: np.ndarray) -> np.ndarray:
        length = traj.shape[0]
        times = np.arange(0, length)
        alpha = (length - 1 - self._shaper.delay_count) / (length - 1)

        traj_fun = []
        for i in range(traj.shape[1]):
            traj_fun.append(sp.interpolate.splrep(times, traj[:, i]))

        scaled_traj = np.zeros_like(traj)
        for i in range(length):
            if times[i] <= alpha * (length - 1):
                for j in range(traj.shape[1]):
                    scaled_traj[i, j] = sp.interpolate.splev(times[i] / alpha, traj_fun[j])
            else:
                scaled_traj[i, :] = traj[-1, :]

        return self._shaper.shape(scaled_traj)[: length, :]
