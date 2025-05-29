import copy

from spatialmath import SE3

from .geometry3d import Geometry3D


class Plane(Geometry3D):
    def __init__(self, base: SE3) -> None:
        super().__init__(base)

    def get_normal_vector(self):
        return copy.deepcopy(self.base.a)
