from typing import List

import numpy as np

from spatialmath import SE3

from .. import Geometry
from ..simplex import Point, UnitVector, Support
from .geometry3d import Geometry3D


class Sphere(Geometry3D, Support):
    def __init__(self, base: SE3, radius: float) -> None:
        super().__init__(base)
        self.radius = radius

    @property
    def points(self) -> List[Geometry]:
        return []

    def calculate_support_point(self, d: UnitVector) -> Point:
        return Point(self.get_t() + self.radius * d.get_t())

    def plot(self, ax, c=None):
        phi, theta = np.mgrid[0:np.pi:20j, 0:2 * np.pi:20j]
        x = self.radius * np.cos(theta) * np.sin(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(phi)

        coordinates = self.calculate_coordinates(x, y, z)
        ax.plot_surface(*coordinates, color='b' if c is None else c, alpha=0.5)
