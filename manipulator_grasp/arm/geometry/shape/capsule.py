from typing import List

import numpy as np
from spatialmath import SE3

from .. import Geometry
from ..simplex import Point, UnitVector, Support
from .geometry3d import Geometry3D


class Capsule(Geometry3D, Support):
    def __init__(self, base: SE3, radius: float, length: float) -> None:
        super().__init__(base)
        self.radius = radius
        self.length = length

    @property
    def points(self) -> List[Geometry]:
        return []

    def calculate_support_point(self, d: UnitVector) -> Point:
        if np.dot(d.get_t(), self.base.a) > 0.0:
            return Point(np.squeeze(self.base * np.array([0, 0, self.length * 0.5])) + d.get_t() * self.radius)
        return Point(np.squeeze(self.base * np.array([0, 0, -self.length * 0.5])) + d.get_t() * self.radius)

    def plot(self, ax, c=None):
        num1 = 20
        num2 = 20

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
        x = self.radius * np.cos(theta) * np.sin(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(phi) + self.length * 0.5
        coordinates = self.calculate_coordinates(x, y, z)
        ax.plot_surface(*coordinates, alpha=0.5, color='blue' if c is None else c)

        x = self.radius * np.cos(theta) * np.sin(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = -self.radius * np.cos(phi) - self.length * 0.5
        coordinates = self.calculate_coordinates(x, y, z)
        ax.plot_surface(*coordinates, alpha=0.5, color='blue' if c is None else c)
