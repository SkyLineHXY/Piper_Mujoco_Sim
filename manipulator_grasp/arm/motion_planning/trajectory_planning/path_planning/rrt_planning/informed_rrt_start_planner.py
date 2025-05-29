import multiprocessing
import random

import numpy as np
from spatialmath import SE3, SO3

from arm.geometry import Ellipsoid

from .node import Node
from .rrt_map import RRTMap
from .rrt_parameter import RRTParameter
from .rrt_star_planner import RRTStarPlanner


class InformedRRTStarPlanner(RRTStarPlanner):
    def __init__(self, rrt_map: RRTMap, rrt_parameter: RRTParameter, pool: multiprocessing.Pool = None) -> None:
        self.c_min = np.linalg.norm(rrt_parameter.goal - rrt_parameter.start)
        self.x_center = (rrt_parameter.goal + rrt_parameter.start) / 2.0

        self.dimension = self.x_center.size
        id1_t = np.zeros_like(self.x_center)
        id1_t[0] = 1.0
        a1 = (rrt_parameter.goal - rrt_parameter.start).reshape(self.dimension, 1) / self.c_min
        m = a1 @ id1_t.reshape(1, self.dimension)
        u, s, vh = np.linalg.svd(m, True, True)
        ones = np.ones_like(self.x_center)
        ones[-1] = np.linalg.det(u) * np.linalg.det(np.transpose(vh))
        ones_diag = np.diag(ones)
        self.c = np.dot(np.dot(u, ones_diag), vh)
        self.r = np.zeros_like(self.x_center)

        super().__init__(rrt_map, rrt_parameter, pool)

    def sample(self) -> Node:
        if not self.success:
            return super().sample()

        L = np.diag(self.r)
        while True:
            x_ball = self.sample_unit_ball()
            rnd = np.squeeze(np.dot(np.dot(self.c, L), x_ball)) + self.x_center
            if self.in_area(rnd):
                return Node(rnd)

    def sample_unit_ball(self) -> np.ndarray:

        radius = random.random()

        thetas = np.random.random(self.dimension - 1) * np.pi
        thetas[-1] *= 2

        xs = np.ones_like(self.x_center)
        for i, theta in enumerate(thetas):
            xs[i + 1] = xs[i] * np.sin(theta)
        for i, theta in enumerate(thetas):
            xs[i] *= np.cos(theta)

        return xs.reshape(self.dimension, 1) * radius

    def update_parameter(self):
        super().update_parameter()
        self.r[0] = self._path_length / 2.0
        self.r[1:] = np.sqrt(self._path_length ** 2 - self.c_min ** 2) / 2.0

    def draw_others(self, ax) -> None:
        if not self.success:
            return
        ellipsoid = Ellipsoid(SE3(*self.x_center) * SE3(SO3(self.c)), self.r)
        ellipsoid.plot(ax, c='y')
