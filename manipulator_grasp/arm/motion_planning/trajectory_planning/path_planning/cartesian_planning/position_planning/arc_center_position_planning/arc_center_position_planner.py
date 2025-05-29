import numpy as np
from spatialmath import SE3

from arm.interface import ModeEnum
from ..position_planning_mode_enum import PositionPlanningModeEnum
from ..position_planner_strategy import PositionPlannerStrategy

from .arc_center_position_parameter import ArcCenterPositionParameter


class ArcCenterPositionPlanner(PositionPlannerStrategy):

    def __init__(self, parameter: ArcCenterPositionParameter):
        super().__init__(parameter)

        self.radius = 0.0
        self.theta = 0.0
        self.T = SE3()

        self.plan()

    @classmethod
    def mode(cls) -> ModeEnum:
        return PositionPlanningModeEnum.ARC_CENTER

    def plan(self) -> None:
        vec_pc_p0 = self.parameter.get_t0() - self.parameter.get_tc()
        vec_pc_p1 = self.parameter.get_t1() - self.parameter.get_tc()

        norm_pc_p0 = np.linalg.norm(vec_pc_p0)
        norm_pc_p1 = np.linalg.norm(vec_pc_p1)

        vec_norm_pc_p0 = vec_pc_p0 / norm_pc_p0
        vec_norm_pc_p1 = vec_pc_p1 / norm_pc_p1

        self.radius = np.linalg.norm(vec_pc_p0)
        self.theta = np.arccos(np.dot(vec_norm_pc_p0, vec_norm_pc_p1))

        vec_cx = vec_norm_pc_p0
        vec_cz = np.cross(vec_norm_pc_p0, vec_norm_pc_p1)
        vec_cz = vec_cz / np.linalg.norm(vec_cz)
        vec_cy = np.cross(vec_cz, vec_cx)

        T = np.eye(4)
        T[:3, :] = np.transpose(np.vstack((vec_cx, vec_cy, vec_cz, self.parameter.get_tc())))
        self.T = SE3(T)

    def interpolate(self, s) -> np.ndarray:
        theta_s = s * self.theta
        x = self.radius * np.cos(theta_s)
        y = self.radius * np.sin(theta_s)

        p = SE3(x=x, y=y, z=0)
        tp = self.T * p
        return tp.t
