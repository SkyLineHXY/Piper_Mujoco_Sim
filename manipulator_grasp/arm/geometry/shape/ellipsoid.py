import copy

import numpy as np
from spatialmath import SE3

from .geometry3d import Geometry3D


class Ellipsoid(Geometry3D):
    def __init__(self, base: SE3, dimensions: np.ndarray) -> None:
        super().__init__(base)
        self.__dimensions = copy.deepcopy(dimensions)

    def plot(self, ax, c=None):
        num1 = 20
        num2 = 20

        theta = np.linspace(0, 2 * np.pi, num1)
        phi = np.linspace(0, np.pi, num2)
        theta, phi = np.meshgrid(theta, phi)

        x = self.__dimensions[0] * np.cos(theta) * np.sin(phi)
        y = self.__dimensions[1] * np.sin(theta) * np.sin(phi)
        z = self.__dimensions[2] * np.cos(phi)

        coordinates = self.calculate_coordinates(x, y, z)
        ax.plot_surface(*coordinates, color='b' if c is None else c, alpha=0.5)
