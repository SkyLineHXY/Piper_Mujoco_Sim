import copy
from abc import ABC

import numpy as np
from spatialmath import SE3


class Geometry3D(ABC):

    def __init__(self, base: SE3) -> None:
        super().__init__()
        self.__base = copy.deepcopy(base)

    @property
    def base(self) -> SE3:
        return copy.deepcopy(self.__base)

    @base.setter
    def base(self, base: SE3):
        self.__base = copy.deepcopy(base)

    def get_t(self):
        return copy.deepcopy(self.__base.t)

    def plot(self, ax, c=None):
        pass

    def calculate_coordinates(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        x_new = np.zeros_like(x)
        y_new = np.zeros_like(y)
        z_new = np.zeros_like(z)

        num1, num2 = x.shape

        for i in range(num1):
            for j in range(num2):
                t = np.squeeze(self.base * [x[i, j], y[i, j], z[i, j]])
                [x_new[i, j], y_new[i, j], z_new[i, j]] = t
        return x_new, y_new, z_new
