from typing import List

import numpy as np
from spatialmath import SE3

from ..simplex import Point, UnitVector, Support
from .geometry3d import Geometry3D
from .circle import Circle


class Cylinder(Geometry3D, Support):

    def __init__(self, base: SE3, radius: float, length: float) -> None:
        super().__init__(base)
        self.radius = radius
        self.length = length

    @property
    def circles(self) -> List[Circle]:
        return [Circle(self.base * SE3.Tz(-self.length * 0.5), self.radius),
                Circle(self.base * SE3.Tz(self.length * 0.5), self.radius)]

    @property
    def points(self) -> List[Point]:
        return []

    def calculate_support_point(self, d: UnitVector) -> Point:
        if np.dot(self.base.a, d.get_t()) < 0.0:
            return self.circles[0].calculate_support_point(d)
        return self.circles[1].calculate_support_point(d)

    def plot(self, ax, c=None):
        num1 = 20
        num2 = 2

        theta = np.linspace(0, 2 * np.pi, num1)
        z = np.linspace(-0.5 * self.length, 0.5 * self.length, num2)
        theta, z = np.meshgrid(theta, z)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)

        coordinates = self.calculate_coordinates(x, y, z)
        ax.plot_surface(*coordinates, alpha=0.5, color='blue' if c is None else c)

        theta = np.linspace(0, 2 * np.pi, num1)
        phi = np.linspace(0, np.pi / 2, num2)
        theta, phi = np.meshgrid(theta, phi)

        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = -0.5 * self.length * np.ones_like(x)
        coordinates = self.calculate_coordinates(x, y, z)
        ax.plot_surface(*coordinates, alpha=0.5, color='blue' if c is None else c)

        z = 0.5 * self.length * np.ones_like(x)
        coordinates = self.calculate_coordinates(x, y, z)
        ax.plot_surface(*coordinates, alpha=0.5, color='blue' if c is None else c)


