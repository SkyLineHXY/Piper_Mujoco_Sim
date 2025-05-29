import numpy as np
from spatialmath import SE3

from arm.interface import ModeEnum
from arm.utils import MathUtils
from ..position_planning_mode_enum import PositionPlanningModeEnum
from ..position_planner_strategy import PositionPlannerStrategy

from .arc_point_position_parameter import ArcPointPositionParameter


class ArcCenterPositionPlanner(PositionPlannerStrategy):

    def __init__(self, parameter: ArcPointPositionParameter):
        super().__init__(parameter)

        self.radius = 0.0
        self.theta = 0.0
        self.flag = 1.0
        self.T = SE3()

        self.plan()

    @classmethod
    def mode(cls) -> ModeEnum:
        return PositionPlanningModeEnum.ARC_POINT

    def plan(self) -> None:
        t0 = self.parameter.get_t0()
        t1 = self.parameter.get_t1()
        tc = self.parameter.get_tc()
        A1 = (t0[1] - t1[1]) * (tc[2] - t1[2]) - (tc[1] - t1[1]) * (t0[2] - t1[2])
        B1 = (tc[0] - t1[0]) * (t0[2] - t1[2]) - (t0[0] - t1[0]) * (tc[2] - t1[2])
        C1 = (t0[0] - t1[0]) * (tc[1] - t1[1]) - (tc[0] - t1[0]) * (t0[1] - t1[1])
        ABC1 = np.array([A1, B1, C1])
        D1 = -(A1 * t1[0] + B1 * t1[1] + C1 * t1[2])

        ABC2 = tc - t0
        D2 = (np.sum(np.power(t0, 2) - np.power(tc, 2))) / 2

        ABC3 = t1 - tc
        D3 = (np.sum(np.power(tc, 2) - np.power(t1, 2))) / 2

        M = np.vstack((ABC1, ABC2, ABC3))
        B = np.hstack((-D1, -D2, -D3))

        assert not MathUtils.near_zero(np.linalg.det(M)), "Unable to form a circle"

        center = np.linalg.inv(M) @ B
        self.radius = np.linalg.norm(t0 - center)

        a = ABC1 / np.linalg.norm(ABC1)

        n = (t0 - center) / np.linalg.norm(t0 - center)

        o = np.cross(a, n)

        T = np.eye(4)
        T[:3, :] = np.transpose(np.vstack((n, o, a, center)))
        self.T = SE3(T)

        invT = np.linalg.inv(T)
        tc_new = invT @ np.hstack((tc, 1))
        t1_new = invT @ np.hstack((t1, 1))

        angle_tc = np.arctan2(tc_new[1], tc_new[0])
        if angle_tc < 0.0:
            angle_tc += 2 * np.pi

        angle_t1 = np.arctan2(t1_new[1], t1_new[0])
        if angle_t1 < 0.0:
            angle_t1 += 2 * np.pi

        self.flag = 1
        self.theta = angle_t1
        if angle_tc > angle_t1:
            self.flag = -1
            self.theta = 2 * np.pi - self.theta

    def interpolate(self, s) -> np.ndarray:
        theta_s = s * self.theta
        x = self.flag * self.radius * np.cos(theta_s)
        y = self.flag * self.radius * np.sin(theta_s)

        p = SE3(x=x, y=y, z=0)
        tp = self.T * p
        return tp.t
