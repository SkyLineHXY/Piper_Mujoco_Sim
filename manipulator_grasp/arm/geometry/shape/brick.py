import copy
from typing import List

import numpy as np
from spatialmath import SE3

from ..simplex import Point, LineSegment, Support
from .geometry3d import Geometry3D
from .plane import Plane


class Brick(Geometry3D, Support):
    def __init__(self, base: SE3, dimensions: np.ndarray) -> None:
        super().__init__(base)
        self.__dimensions = copy.deepcopy(dimensions)

    @property
    def dimensions(self):
        return copy.deepcopy(self.__dimensions)

    @property
    def points(self) -> List[Point]:
        return [Point(
            (self.base * SE3.Trans(
                *(self.__dimensions * np.array([i % 2 - 0.5, i % 4 // 2 - 0.5, i // 4 - 0.5])))).t
        ) for i in range(8)]

    @property
    def xn_yn_zn_point(self) -> Point:
        return Point(np.squeeze(self.base * (self.__dimensions * np.array([-1, -1, -1]) * 0.5)))

    @property
    def xp_yn_zn_point(self) -> Point:
        return Point(np.squeeze(self.base * (self.__dimensions * np.array([1, -1, -1]) * 0.5)))

    @property
    def xn_yp_zn_point(self) -> Point:
        return Point(np.squeeze(self.base * (self.__dimensions * np.array([-1, 1, -1]) * 0.5)))

    @property
    def xp_yp_zn_point(self) -> Point:
        return Point(np.squeeze(self.base * (self.__dimensions * np.array([1, 1, -1]) * 0.5)))

    @property
    def xn_yn_zp_point(self) -> Point:
        return Point(np.squeeze(self.base * (self.__dimensions * np.array([-1, -1, 1]) * 0.5)))

    @property
    def xp_yn_zp_point(self) -> Point:
        return Point(np.squeeze(self.base * (self.__dimensions * np.array([1, -1, 1]) * 0.5)))

    @property
    def xn_yp_zp_point(self) -> Point:
        return Point(np.squeeze(self.base * (self.__dimensions * np.array([-1, 1, 1]) * 0.5)))

    @property
    def xp_yp_zp_point(self) -> Point:
        return Point(np.squeeze(self.base * (self.__dimensions * np.array([1, 1, 1]) * 0.5)))

    @property
    def yn_zn_line_segment(self) -> LineSegment:
        return LineSegment(self.xn_yn_zn_point, self.xp_yn_zn_point)

    @property
    def yp_zn_line_segment(self) -> LineSegment:
        return LineSegment(self.xn_yp_zn_point, self.xp_yp_zn_point)

    @property
    def yn_zp_line_segment(self) -> LineSegment:
        return LineSegment(self.xn_yn_zp_point, self.xp_yn_zp_point)

    @property
    def yp_zp_line_segment(self) -> LineSegment:
        return LineSegment(self.xn_yp_zp_point, self.xp_yp_zp_point)

    @property
    def xn_zn_line_segment(self) -> LineSegment:
        return LineSegment(self.xn_yn_zn_point, self.xn_yp_zn_point)

    @property
    def xp_zn_line_segment(self) -> LineSegment:
        return LineSegment(self.xp_yn_zn_point, self.xp_yp_zn_point)

    @property
    def xn_zp_line_segment(self) -> LineSegment:
        return LineSegment(self.xn_yn_zp_point, self.xn_yp_zp_point)

    @property
    def xp_zp_line_segment(self) -> LineSegment:
        return LineSegment(self.xp_yn_zp_point, self.xp_yp_zp_point)

    @property
    def xn_yn_line_segment(self) -> LineSegment:
        return LineSegment(self.xn_yn_zn_point, self.xn_yn_zp_point)

    @property
    def xp_yn_line_segment(self) -> LineSegment:
        return LineSegment(self.xp_yn_zn_point, self.xp_yn_zp_point)

    @property
    def xn_yp_line_segment(self) -> LineSegment:
        return LineSegment(self.xn_yp_zn_point, self.xn_yp_zp_point)

    @property
    def xp_yp_line_segment(self) -> LineSegment:
        return LineSegment(self.xp_yp_zn_point, self.xp_yp_zp_point)

    @property
    def xn_plane(self) -> Plane:
        center = self.get_t() + np.array(self.__dimensions * np.array([-1.0, 0.0, 0.0])) * 0.5
        return Plane(SE3.Trans(*center) * SE3.Ry(-np.pi / 2))

    @property
    def xp_plane(self) -> Plane:
        center = self.get_t() + np.array(self.__dimensions * np.array([1.0, 0.0, 0.0])) * 0.5
        return Plane(SE3.Trans(*center) * SE3.Ry(np.pi / 2))

    @property
    def yn_plane(self) -> Plane:
        center = self.get_t() + np.array(self.__dimensions * np.array([0.0, -1.0, 0.0])) * 0.5
        return Plane(SE3.Trans(*center) * SE3.Rx(np.pi / 2))

    @property
    def yp_plane(self) -> Plane:
        center = self.get_t() + np.array(self.__dimensions * np.array([0.0, -1.0, 0.0])) * 0.5
        return Plane(SE3.Trans(*center) * SE3.Rx(-np.pi / 2))

    @property
    def zn_plane(self) -> Plane:
        center = self.get_t() + np.array(self.__dimensions * np.array([0.0, 0.0, -1.0])) * 0.5
        return Plane(SE3.Trans(*center) * SE3.Rx(np.pi))

    @property
    def zp_plane(self) -> Plane:
        center = self.get_t() + np.array(self.__dimensions * np.array([0.0, 0.0, 1.0])) * 0.5
        return Plane(SE3.Trans(*center))

    def plot(self, ax, c=None):
        planes = [
            [self.xn_zn_line_segment, self.xn_zp_line_segment],
            [self.xp_zn_line_segment, self.xp_zp_line_segment],
            [self.yn_zn_line_segment, self.yn_zp_line_segment],
            [self.yp_zn_line_segment, self.yp_zp_line_segment],
            [self.xn_zn_line_segment, self.xp_zn_line_segment],
            [self.xn_zp_line_segment, self.xp_zp_line_segment]
        ]

        for plane in planes:
            points = [
                np.array([
                    [plane[0].get_point0().get_t()[i], plane[0].get_point1().get_t()[i]],
                    [plane[1].get_point0().get_t()[i], plane[1].get_point1().get_t()[i]]
                ]) for i in range(3)
            ]

            ax.plot_surface(*points, alpha=0.5, color='b' if c is None else c)
