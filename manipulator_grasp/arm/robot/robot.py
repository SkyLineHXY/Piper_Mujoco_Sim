import abc
import copy
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

from typing import List
from manipulator_grasp.arm.geometry import Geometry3D
import modern_robotics as mr


def get_transformation_mdh(alpha, a, d, theta, sigma, q) -> SE3:
    if sigma == 0:
        theta += q
    elif sigma == 1:
        d += q

    return SE3.Rx(alpha) * SE3.Tx(a) * SE3.Tz(d) * SE3.Rz(theta)


def circle(V):
    return np.array([
        [V[0], V[1], V[2], 0, 0, 0, 0, V[5], -V[4], 0],
        [0, V[0], 0, V[1], V[2], 0, -V[5], 0, V[3], 0],
        [0, 0, V[0], 0, V[1], V[2], V[4], -V[3], 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -V[2], V[1], V[3]],
        [0, 0, 0, 0, 0, 0, V[2], 0, -V[0], V[4]],
        [0, 0, 0, 0, 0, 0, -V[1], V[0], 0, V[5]]
    ])


def wrap(theta) -> tuple:
    number = np.floor(theta / (2 * np.pi))
    theta -= 2 * np.pi * number
    if theta < -np.pi:
        theta += 2 * np.pi
        number -= 1
    elif theta > np.pi:
        theta -= 2 * np.pi
        number += 1

    return theta, number


class Robot(abc.ABC):
    _link_para_num = 10

    def __init__(self) -> None:
        super().__init__()
        self.robot: rtb.Robot = None
        self._dof = 0
        self.q0 = [0.0 for _ in range(self.dof)]
        self.alpha_array = [0.0 for _ in range(self.dof)]
        self.a_array = [0.0 for _ in range(self.dof)]
        self.d_array = [0.0 for _ in range(self.dof)]
        self.theta_array = [0.0 for _ in range(self.dof)]
        self.sigma_array = [0 for _ in range(self.dof)]
        self._g = np.array([0, 0, -9.81])
        self._phi = 0.0

        self._Ms = []
        self._Ses = []
        self._Gs = []
        self._Jms = []

        self._base = SE3()
        self._tool = SE3()

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def inertial_parameters(self) -> np.ndarray:
        parameters = []
        for i in range(self._dof):
            parameter = [self._Gs[i][0, 0], self._Gs[i][0, 1], self._Gs[i][0, 2], self._Gs[i][1, 1], self._Gs[i][1, 2],
                         self._Gs[i][2, 2], self._Gs[i][2, 4], self._Gs[i][0, 5], self._Gs[i][1, 3], self._Gs[i][3, 3]]
            parameters.extend(parameter)
        parameters.extend(self._Jms)
        return np.array(parameters)

    @inertial_parameters.setter
    def inertial_parameters(self, parameters):
        for i in range(self._dof):
            I = np.array([
                [parameters[i * Robot._link_para_num + 0], parameters[i * Robot._link_para_num + 1],
                 parameters[i * Robot._link_para_num + 2]],
                [parameters[i * Robot._link_para_num + 1], parameters[i * Robot._link_para_num + 3],
                 parameters[i * Robot._link_para_num + 4]],
                [parameters[i * Robot._link_para_num + 2], parameters[i * Robot._link_para_num + 4],
                 parameters[i * Robot._link_para_num + 5]]
            ])
            mp = np.array([parameters[i * Robot._link_para_num + 6], parameters[i * Robot._link_para_num + 7],
                           parameters[i * Robot._link_para_num + 8]])
            mp_mat = mr.VecToso3(mp)
            m_mat = parameters[i * Robot._link_para_num + 9] * np.eye(3)

            self._Gs[i][:3, :3] = I
            self._Gs[i][:3, 3:] = mp_mat
            self._Gs[i][3:, :3] = mp_mat.T
            self._Gs[i][3:, 3:] = m_mat
        for i in range(self._dof):
            self._Jms[i] = parameters[self._dof * Robot._link_para_num + i]

    def fkine(self, q) -> SE3:
        """
        计算机器人在给定关节角度q下的正向运动学结果。

        参数:
        q: 关节角度的列表或数组

        返回值:
        返回一个SE3对象，表示末端执行器在世界坐标系中的位姿。
        """
        return self.robot.fkine(q)

    def get_cartesian(self):
        """
        获取当前机器人配置下末端执行器的笛卡尔坐标。

        返回值:
        返回一个SE3对象，表示末端执行器的当前位姿。
        """
        return self.fkine(self.q0)

    def move_joint(self, q):
        """
        移动机器人到指定的关节空间位置。

        参数:
        q: 目标关节角度的列表或数组
        """
        self.set_joint(q)

    def set_joint(self, q):
        """
        设置机器人的关节角度。

        参数:
        q: 新的关节角度列表或数组
        """
        self.q0 = q[:]
        self.set_robot_config(self.q0)

    def set_robot_config(self, q):
        """
        根据给定的关节角度设置机器人配置（具体实现由子类提供）。

        参数:
        q: 关节角度列表或数组
        """
        pass

    def move_cartesian(self, T: SE3):
        """
        移动机器人到指定的笛卡尔空间位姿。

        参数:
        T: 目标位姿，表示为SE3对象

        异常:
        如果逆运动学求解失败，将抛出断言异常。
        """
        q = self.ikine(T)

        assert len(q)  # inverse kinematics failure
        self.q0 = q[:]

    @abc.abstractmethod
    def ikine(self, Tep: SE3) -> np.ndarray:
        pass

    def get_joint(self):
        return copy.deepcopy(self.q0)

    def get_geometries(self) -> List[Geometry3D]:
        pass

    def get_inertia(self, q: np.ndarray) -> np.ndarray:
        return self.robot.inertia(q)

    def get_coriolis(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        return self.robot.coriolis(q, dq)

    def get_gravity(self, q: np.ndarray):
        return self.robot.gravload(q)

    def inv_dynamics(self, qs, dqs, ddqs) -> np.ndarray:
        return self.robot.rne(qs, dqs, ddqs)

    def get_identification_matrix(self, qs, dqs, ddqs):
        Mi = np.eye(4)
        Ai = np.zeros((6, self._dof))
        AdTi = [[None]] * (self._dof + 1)
        Vi = np.zeros((6, self._dof + 1))
        Vdi = np.zeros((6, self._dof + 1))
        Vdi[:, 0] = np.r_[[0, 0, 0], -self._g]
        AdTi[self._dof] = mr.Adjoint(mr.TransInv(self._Ms[self._dof]))
        for i in range(self._dof):
            Mi = np.dot(Mi, self._Ms[i])
            Ai[:, i] = np.dot(mr.Adjoint(mr.TransInv(Mi)), self._Ses[i])
            AdTi[i] = mr.Adjoint(np.dot(mr.MatrixExp6(mr.VecTose3(Ai[:, i] * -qs[i])), mr.TransInv(self._Ms[i])))
            Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dqs[i]
            Vdi[:, i + 1] = np.dot(AdTi[i], Vdi[:, i]) + Ai[:, i] * ddqs[i] + np.dot(mr.ad(Vi[:, i + 1]), Ai[:, i]) * \
                            dqs[i]

        Yi = np.zeros((6, Robot._link_para_num * self._dof))
        Y = np.zeros((self.dof, Robot._link_para_num * self._dof))
        for i in range(self._dof - 1, -1, -1):
            Yi = AdTi[i + 1].T @ Yi
            Yi[:, i * Robot._link_para_num: (i + 1) * Robot._link_para_num] = circle(Vdi[:, i + 1]) - mr.ad(
                Vi[:, i + 1]).T @ circle(Vi[:, i + 1])
            Y[i, :] = Ai[:, i].T @ Yi
        Y = np.hstack((Y, np.diag(ddqs)))
        return Y

    def get_adaptive_identification_matrix(self, qs, dqs, dqrs, ddqrs):
        Mi = np.eye(4)
        Ai = np.zeros((6, self._dof))
        AdTi = [[None]] * (self._dof + 1)
        Vi = np.zeros((6, self._dof + 1))
        Vri = np.zeros((6, self._dof + 1))
        Vdri = np.zeros((6, self._dof + 1))
        Vdri[:, 0] = np.r_[[0, 0, 0], -self._g]
        AdTi[self._dof] = mr.Adjoint(mr.TransInv(self._Ms[self._dof]))
        for i in range(self._dof):
            Mi = np.dot(Mi, self._Ms[i])
            Ai[:, i] = np.dot(mr.Adjoint(mr.TransInv(Mi)), self._Ses[i])
            AdTi[i] = mr.Adjoint(np.dot(mr.MatrixExp6(mr.VecTose3(Ai[:, i] * -qs[i])), mr.TransInv(self._Ms[i])))
            Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dqs[i]
            Vri[:, i + 1] = np.dot(AdTi[i], Vri[:, i]) + Ai[:, i] * dqrs[i]
            Vdri[:, i + 1] = np.dot(AdTi[i], Vdri[:, i]) + Ai[:, i] * ddqrs[i] + np.dot(mr.ad(Vi[:, i + 1]), Ai[:, i]) * \
                             dqrs[i]

        Yi = np.zeros((6, Robot._link_para_num * self._dof))
        Y = np.zeros((self._dof, Robot._link_para_num * self._dof))
        for i in range(self._dof - 1, -1, -1):
            Yi = AdTi[i + 1].T @ Yi
            Yi[:, i * Robot._link_para_num: (i + 1) * Robot._link_para_num] = circle(Vdri[:, i + 1]) - 0.5 * mr.ad(
                Vi[:, i + 1]).T @ circle(Vri[:, i + 1]) - 0.5 * mr.ad(Vri[:, i + 1]).T @ circle(
                Vi[:, i + 1]) + 0.5 * circle(mr.ad(Vri[:, i + 1]) @ Vi[:, i + 1])
            Y[i, :] = Ai[:, i].T @ Yi
        Y = np.hstack((Y, np.diag(ddqrs)))
        return Y

    def inv_dynamics_adaptive(self, qs, dqs, dqrs, ddqrs):
        Mi = np.eye(4)
        Ai = np.zeros((6, self._dof))
        AdTi = [[None]] * (self._dof + 1)
        Vi = np.zeros((6, self._dof + 1))
        Vri = np.zeros((6, self._dof + 1))
        Vdri = np.zeros((6, self._dof + 1))
        Vdri[:, 0] = np.r_[[0, 0, 0], -self._g]
        AdTi[self._dof] = mr.Adjoint(mr.TransInv(self._Ms[self._dof]))
        Fi = np.zeros(6)
        taulist = np.zeros(self._dof)
        for i in range(self._dof):
            Mi = np.dot(Mi, self._Ms[i])
            Ai[:, i] = np.dot(mr.Adjoint(mr.TransInv(Mi)), self._Ses[i])
            AdTi[i] = mr.Adjoint(np.dot(mr.MatrixExp6(mr.VecTose3(Ai[:, i] * -qs[i])), mr.TransInv(self._Ms[i])))
            Vi[:, i + 1] = np.dot(AdTi[i], Vi[:, i]) + Ai[:, i] * dqs[i]
            Vri[:, i + 1] = np.dot(AdTi[i], Vri[:, i]) + Ai[:, i] * dqrs[i]
            Vdri[:, i + 1] = np.dot(AdTi[i], Vdri[:, i]) + Ai[:, i] * ddqrs[i] + np.dot(mr.ad(Vi[:, i + 1]), Ai[:, i]) * \
                             dqrs[i]

        for i in range(self._dof - 1, -1, -1):
            Fi = np.array(AdTi[i + 1]).T @ Fi + self._Gs[i] @ Vdri[:, i + 1] + 0.5 * (
                    -mr.ad(Vi[:, i + 1]).T @ (self._Gs[i] @ Vri[:, i + 1])
                    - mr.ad(Vri[:, i + 1]).T @ (self._Gs[i] @ Vi[:, i + 1])
                    + self._Gs[i] @ (mr.ad(Vri[:, i + 1]) @ Vi[:, i + 1])
            )

            taulist[i] = np.dot(np.array(Fi).T, Ai[:, i]) + self._Jms[i] * ddqrs[i]

        return taulist

    def set_tool(self, tool: SE3):
        self._tool = tool
        self.robot.tool = self._tool

    def disable_tool(self):
        self._tool = SE3()
        self.robot.tool = self._tool

    def set_base(self, base: np.ndarray):
        self._base = SE3.Trans(base)
        self.robot.base = self._base

    def disable_base(self):
        self._base = SE3()
        self.robot.base = self._base

    @property
    def base(self):
        return self._base

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi: float):
        self._phi = phi

    def __getstate__(self):
        state = {"dof": self._dof,
                 "q0": self.q0,
                 "alpha_array": self.alpha_array,
                 "a_array": self.a_array,
                 "d_array": self.d_array,
                 "theta_array": self.theta_array,
                 }
        return state

    def __setstate__(self, state):
        self._dof = state["dof"]
        self.q0 = state["q0"]
        self.alpha_array = state["alpha_array"]
        self.a_array = state["a_array"]
        self.d_array = state["d_array"]
        self.theta_array = state["theta_array"]
        links = []
        for i in range(6):
            links.append(
                rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], offset=self.theta_array[i],
                           mdh=True))
        self.robot = rtb.DHRobot(links)
