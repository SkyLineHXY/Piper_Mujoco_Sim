import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3, SO3, UnitQuaternion
import modern_robotics as mr

from .robot import Robot, get_transformation_mdh, wrap
from .robot_config import RobotConfig


class IIWA14(Robot):

    def __init__(self) -> None:
        super().__init__()

        self.d1 = 0.1575 + 0.2025
        self.d3 = 0.2045 + 0.2155
        self.d5 = 0.1845 + 0.2155
        self.d7 = 0.081 + 0.045

        self._dof = 7
        self.q0 = [0.0 for _ in range(self._dof)]

        self.alpha_array = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2]
        self.a_array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.d_array = [self.d1, 0.0, self.d3, 0.0, self.d5, 0.0, self.d7]
        self.theta_array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.sigma_array = [0, 0, 0, 0, 0, 0, 0]

        m1 = 5.76
        r1 = np.array([0.0, -0.03, -0.2025 + 0.12])
        Ixx1 = 0.0333
        Iyy1 = 0.033
        Izz1 = 0.0123
        I1 = np.diag([Ixx1, Iyy1, Izz1])
        R1 = SO3().R

        m2 = 6.35
        r2 = np.array([-0.0003, -0.059, 0.042])
        Ixx2 = 0.0305
        Iyy2 = 0.0304
        Izz2 = 0.011
        I2 = np.diag([Ixx2, Iyy2, Izz2])
        R2 = SO3.Rx(np.pi / 2).R

        m3 = 3.5
        r3 = np.array([0.0, 0.03, -0.2155 + 0.13])
        Ixx3 = 0.025
        Iyy3 = 0.0238
        Izz3 = 0.0076
        I3 = np.diag([Ixx3, Iyy3, Izz3])
        R3 = SO3().R

        m4 = 3.5
        r4 = np.array([0.0, 0.067, 0.034])
        Ixx4 = 0.017
        Iyy4 = 0.0164
        Izz4 = 0.006
        I4 = np.diag([Ixx4, Iyy4, Izz4])
        R4 = UnitQuaternion([1, 1, 0, 0]).R

        m5 = 3.5
        r5 = np.array([-0.0001, -0.021, -0.2155 + 0.076])
        Ixx5 = 0.01
        Iyy5 = 0.0087
        Izz5 = 0.00449
        I5 = np.diag([Ixx5, Iyy5, Izz5])
        R5 = SE3.Rz(np.pi).R

        m6 = 1.8
        r6 = np.array([0.0, -0.0006, 0.0004])
        Ixx6 = 0.0049
        Iyy6 = 0.0047
        Izz6 = 0.0036
        I6 = np.diag([Ixx6, Iyy6, Izz6])
        R6 = UnitQuaternion([0, 0, -1, -1]).R

        m7 = 1.2
        r7 = np.array([0.0, 0.0, -0.045 + 0.02])
        Ixx7 = 0.001
        Iyy7 = 0.001
        Izz7 = 0.001
        I7 = np.diag([Ixx7, Iyy7, Izz7])
        R7 = SO3().R

        ms = [m1, m2, m3, m4, m5, m6, m7]
        rs = [r1, r2, r3, r4, r5, r6, r7]
        Is = [I1, I2, I3, I4, I5, I6, I7]
        Rs = [R1, R2, R3, R4, R5, R6, R7]

        T = SE3()
        self.Slist = np.zeros((6, self._dof))
        self.Glist = []
        self.Jms = []
        self.Mlist = []
        for i in range(self._dof):
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)
            if i == 0:
                self.Mlist.append(T.A @ Ti.A)
            else:
                self.Mlist.append(Ti.A)

            T: SE3 = T * Ti
            self.Slist[:, i] = np.hstack((T.a, np.cross(T.t, T.a)))

            Gm = np.zeros((6, 6))
            Gm[:3, :3] = Is[i]
            Gm[3:, 3:] = ms[i] * np.eye(3)
            Tab = mr.RpToTrans(Rs[i], rs[i])
            Tba = mr.TransInv(Tab)
            AdT = mr.Adjoint(Tba)
            self.Glist.append(AdT.T @ Gm @ AdT)

        self.M = T.A
        self.Mlist.append(SE3().A)

        links = []
        for i in range(self._dof):
            links.append(
                rtb.DHLink(d=self.d_array[i], alpha=self.alpha_array[i], a=self.a_array[i], offset=self.theta_array[i],
                           mdh=True, m=ms[i], r=rs[i], I=(Rs[i] @ Is[i] @ Rs[i].T)))
            self.robot = rtb.DHRobot(links)

        self.robot_config = RobotConfig()

        self._min_phi = np.zeros((2, 7))
        self._min_phi[0, :] = -np.pi
        self._min_phi[1, :] = np.pi
        self._max_phi = np.zeros((2, 7))
        self._max_phi[0, :] = -np.pi
        self._max_phi[1, :] = np.pi

        self._min_phi = np.zeros((2, 7))
        self._max_phi = np.zeros((2, 7))

        self._phi_limit_all = np.zeros((2, 7))
        self._phi_limit_all[0, 3] = -np.pi
        self._phi_limit_all[1, 3] = np.pi

        self._phi_limit = np.zeros(2)

        self._pivot_joint_index = [0, 2, 4, 6]
        self._hinge_joint_index = [1, 5]

        self._q_lim_low = np.array([-2.96706, -2.0944, -2.96706, -2.0944, -2.96706, -2.0944, -3.05433])
        self._q_lim_up = np.array([2.96706, 2.0944, 2.96706, 2.0944, 2.96709, 2.0944, 3.05433])

        self._pivot_vecs = np.zeros((6, len(self._pivot_joint_index)))
        self._hinge_vecs = np.zeros((3, len(self._hinge_joint_index)))

        self._K = 0.1
        self._alpha = 20.0

        self._filter_factor = 0.95

    @property
    def q_lim_low(self):
        return self._q_lim_low.copy()

    @property
    def q_lim_up(self):
        return self._q_lim_up.copy()

    @property
    def phi_limit(self):
        return self._phi_limit

    @property
    def phi_limit_all(self):
        return self._phi_limit_all

    def ikine(self, Tep: SE3) -> np.ndarray:

        T07: SE3 = self._base.inv() * Tep * self._tool.inv()

        a = T07.a
        t = T07.t

        S = np.array([0, 0, self.d1])
        W = t - self.d7 * a

        L_sw = np.linalg.norm(W - S)
        V_sw = (W - S) / L_sw

        qs = np.zeros(self._dof)

        # solve q4
        q4_condition = np.power(L_sw, 2) - np.power(self.d3, 2) - np.power(self.d5, 2)
        if np.abs(q4_condition) > (2 * self.d3 * self.d5):
            return np.array([])
        if self.robot_config.inline == 1:
            qs[3] = np.arccos(q4_condition / (2 * self.d3 * self.d5))
        elif self.robot_config.inline == -1:
            qs[3] = -np.arccos(q4_condition / (2 * self.d3 * self.d5))
        else:
            raise ValueError("Wrong inline value")

        x = (L_sw * L_sw + self.d3 * self.d3 - self.d5 * self.d5) / (2 * L_sw)
        r = np.sqrt(self.d3 * self.d3 - x * x)

        F = S + x * V_sw

        L_fe = np.array([(- V_sw[0] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                         (- V_sw[1] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                         1])

        E = L_fe / np.linalg.norm(L_fe) * r + F

        V_se = (E - S) / np.linalg.norm(E - S)
        V_ew = (W - E) / np.linalg.norm(W - E)

        R30_z = V_se
        R30_y = - np.sign(qs[3]) * np.cross(V_se, V_ew)
        R30_y = R30_y / np.linalg.norm(R30_y)
        R30_x = np.cross(R30_y, R30_z)
        R30 = np.vstack((R30_x, R30_y, R30_z)).T

        u_hat = np.array([[0, -V_sw[2], V_sw[1]],
                          [V_sw[2], 0, -V_sw[0]],
                          [-V_sw[1], V_sw[0], 0]])
        R3 = (np.eye(3) + u_hat * np.sin(self._phi) + u_hat @ u_hat * (1 - np.cos(self._phi))) @ R30

        # solve q2
        if self.robot_config.overhead == 1:
            qs[1] = np.arccos(R3[2, 2])
        elif self.robot_config.overhead == -1:
            qs[1] = -np.arccos(R3[2, 2])
        else:
            raise ValueError("Wrong overhead value")

        qs[0] = np.arctan2(R3[1, 2] / np.sin(qs[1]), R3[0, 2] / np.sin(qs[1]))
        qs[2] = np.arctan2(R3[2, 1] / np.sin(qs[1]), -R3[2, 0] / np.sin(qs[1]))

        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], self.theta_array[0],
                                     self.sigma_array[0], qs[0])
        T12 = get_transformation_mdh(self.alpha_array[1], self.a_array[1], self.d_array[1], self.theta_array[1],
                                     self.sigma_array[1], qs[1])
        T23 = get_transformation_mdh(self.alpha_array[2], self.a_array[2], self.d_array[2], self.theta_array[2],
                                     self.sigma_array[2], qs[2])
        T34 = get_transformation_mdh(self.alpha_array[3], self.a_array[3], self.d_array[3], self.theta_array[3],
                                     self.sigma_array[3], qs[3])

        T04: SE3 = T01 * T12 * T23 * T34
        T47 = (T04.inv() * Tep).A

        # solve q6
        if self.robot_config.wrist == 1:
            qs[5] = np.arccos(T47[1, 2])
        elif self.robot_config.wrist == -1:
            qs[5] = - np.arccos(T47[1, 2])
        else:
            raise ValueError("Wrong wrist configuration")
        qs[4] = np.arctan2(-T47[2, 2] / np.sin(qs[5]), T47[0, 2] / np.sin(qs[5]))
        qs[6] = np.arctan2(T47[1, 1] / np.sin(qs[5]), -T47[1, 0] / np.sin(qs[5]))

        q0_s = list(map(wrap, self.q0))

        for i in range(self._dof):
            if qs[i] - q0_s[i][0] > np.pi:
                qs[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif qs[i] - q0_s[i][0] < -np.pi:
                qs[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                qs[i] += q0_s[i][1] * 2 * np.pi

        return qs

    def set_robot_config(self, q: np.ndarray):
        # overhead
        if wrap(q[1])[0] >= 0:
            self.robot_config.overhead = 1
            self._q_lim_low[1] = 0.15
            self._q_lim_up[1] = 2.0944
        else:
            self.robot_config.overhead = -1
            self._q_lim_low[1] = -2.0944
            self._q_lim_up[1] = -0.15

        # inline
        if wrap(q[3])[0] >= 0:
            self.robot_config.inline = 1
        else:
            self.robot_config.inline = -1

        # wrist
        if wrap(q[5])[0] >= 0:
            self.robot_config.wrist = 1
            self._q_lim_low[5] = 0.15
            self._q_lim_up[5] = 2.0944
        else:
            self.robot_config.wrist = -1
            self._q_lim_low[5] = -2.0944
            self._q_lim_up[5] = -0.15

        # phi
        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], self.theta_array[0],
                                     self.sigma_array[0], q[0])
        T12 = get_transformation_mdh(self.alpha_array[1], self.a_array[1], self.d_array[1], self.theta_array[1],
                                     self.sigma_array[1], q[1])
        T23 = get_transformation_mdh(self.alpha_array[2], self.a_array[2], self.d_array[2], self.theta_array[2],
                                     self.sigma_array[2], q[2])
        T34 = get_transformation_mdh(self.alpha_array[3], self.a_array[3], self.d_array[3], self.theta_array[3],
                                     self.sigma_array[3], q[3])
        T45 = get_transformation_mdh(self.alpha_array[4], self.a_array[4], self.d_array[4], self.theta_array[4],
                                     self.sigma_array[4], q[4])

        T03 = T01 * T12 * T23
        T04 = T03 * T34
        T05 = T04 * T45

        S = T01.t
        W = T05.t

        L_sw = np.linalg.norm(W - S)
        V_sw = (W - S) / L_sw

        x = (L_sw * L_sw + self.d3 * self.d3 - self.d5 * self.d5) / (2 * L_sw)
        r = np.sqrt(np.abs(self.d3 * self.d3 - x * x))

        F = S + x * V_sw

        L_fe = np.array([(- V_sw[0] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                         (- V_sw[1] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                         1])

        E = L_fe / np.linalg.norm(L_fe) * r + F

        V_se = (E - S) / np.linalg.norm(E - S)
        V_ew = (W - E) / np.linalg.norm(W - E)

        R30_z = V_se
        R30_y = -np.sign(q[3]) * np.cross(V_se, V_ew)
        R30_y = R30_y / np.linalg.norm(R30_y)
        R30_x = np.cross(R30_y, R30_z)
        R30 = np.vstack((R30_x, R30_y, R30_z)).T

        R3 = T03.R

        R_phi = R3 @ np.linalg.inv(R30)

        vec = mr.so3ToVec(mr.MatrixLog3(R_phi))
        self._phi = np.sign(np.dot(vec, V_sw)) * np.linalg.norm(vec)

    def move_cartesian(self, T: SE3):
        q = self.ikine(T)

        if q.size != 0:
            self.q0 = q[:]

    def move_cartesian_with_avoidance(self, T: SE3):
        q = self.ikine_with_avoidance(T)
        if q.size != 0:
            self.q0 = q[:]

    def cal_phi_limit(self, Tep: SE3):
        T07: SE3 = Tep * self._tool.inv()

        a = T07.a
        t = T07.t

        S = np.array([0, 0, self.d1])
        W = t - self.d7 * a

        L_sw = np.linalg.norm(W - S)
        V_sw = (W - S) / L_sw

        # solve q4
        q4_condition = np.power(L_sw, 2) - np.power(self.d3, 2) - np.power(self.d5, 2)
        if np.abs(q4_condition) > (2 * self.d3 * self.d5):
            return np.array([])
        q4 = self.robot_config.inline * np.arccos(q4_condition / (2 * self.d3 * self.d5))

        x = (L_sw * L_sw + self.d3 * self.d3 - self.d5 * self.d5) / (2 * L_sw)
        r = np.sqrt(self.d3 * self.d3 - x * x)

        F = S + x * V_sw

        L_fe = np.array([(- V_sw[0] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                         (- V_sw[1] * V_sw[2]) / (V_sw[0] * V_sw[0] + V_sw[1] * V_sw[1]),
                         1])

        E = L_fe / np.linalg.norm(L_fe) * r + F

        V_se = (E - S) / np.linalg.norm(E - S)
        V_ew = (W - E) / np.linalg.norm(W - E)

        R30_z = V_se
        R30_y = - np.sign(q4) * np.cross(V_se, V_ew)
        R30_y = R30_y / np.linalg.norm(R30_y)
        R30_x = np.cross(R30_y, R30_z)
        R30 = np.vstack((R30_x, R30_y, R30_z)).T

        u_hat = np.array([[0, -V_sw[2], V_sw[1]],
                          [V_sw[2], 0, -V_sw[0]],
                          [-V_sw[1], V_sw[0], 0]])

        As = u_hat @ R30
        Bs = -u_hat @ u_hat @ R30
        Cs = (np.eye(3) + u_hat @ u_hat) @ R30

        T34 = get_transformation_mdh(self.alpha_array[3], self.a_array[3], self.d_array[3], self.theta_array[4],
                                     self.sigma_array[4], q4)
        R34 = T34.R
        R07 = T07.R
        Aw = R34.T @ As.T @ R07
        Bw = R34.T @ Bs.T @ R07
        Cw = R34.T @ Cs.T @ R07

        self._pivot_vecs[:, 0] = self.robot_config.overhead * np.array(
            [As[1, 2], Bs[1, 2], Cs[1, 2], As[0, 2], Bs[0, 2], Cs[0, 2]])
        self._pivot_vecs[:, 1] = self.robot_config.overhead * np.array(
            [As[2, 1], Bs[2, 1], Cs[2, 1], -As[2, 0], -Bs[2, 0], -Cs[2, 0]])
        self._pivot_vecs[:, 2] = self.robot_config.wrist * np.array(
            [-Aw[2, 2], -Bw[2, 2], -Cw[2, 2], Aw[0, 2], Bw[0, 2], Cw[0, 2]])
        self._pivot_vecs[:, 3] = self.robot_config.wrist * np.array(
            [Aw[1, 1], Bw[1, 1], Cw[1, 1], -Aw[1, 0], -Bw[1, 0], -Cw[1, 0]])

        for i in range(len(self._pivot_joint_index)):
            self._min_phi[:, self._pivot_joint_index[i]] = self.cal_pivot_phi(self._pivot_vecs[:, i], self._q_lim_low[
                self._pivot_joint_index[i]])
            self._max_phi[:, self._pivot_joint_index[i]] = self.cal_pivot_phi(self._pivot_vecs[:, i], self._q_lim_up[
                self._pivot_joint_index[i]])

        self._hinge_vecs[:, 0] = self.robot_config.overhead * np.array([As[2, 2], Bs[2, 2], Cs[2, 2]])
        self._hinge_vecs[:, 1] = self.robot_config.wrist * np.array([Aw[1, 2], Bw[1, 2], Cw[1, 2]])

        for i in range(len(self._hinge_joint_index)):
            self._min_phi[:, self._hinge_joint_index[i]] = self.cal_hinge_phi(self._hinge_vecs[:, i], self._q_lim_low[
                self._hinge_joint_index[i]])
            self._max_phi[:, self._hinge_joint_index[i]] = self.cal_hinge_phi(self._hinge_vecs[:, i], self._q_lim_up[
                self._hinge_joint_index[i]])

        self.q0[3] = q4

    def cal_pivot_q(self, vec, phi):
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        u = vec[0] * sin_phi + vec[1] * cos_phi + vec[2]
        v = vec[3] * sin_phi + vec[4] * cos_phi + vec[5]
        return np.arctan2(u, v)

    def cal_hinge_q(self, vec, phi):
        return np.arccos(vec[0] * np.sin(phi) + vec[1] * np.cos(phi) + vec[2])

    def cal_pivot_phi(self, vec, q):
        tan_q = np.tan(q)
        ap = (vec[5] - vec[4]) * tan_q + (vec[1] - vec[2])
        bp = 2 * (vec[3] * tan_q - vec[0])
        cp = (vec[4] + vec[5]) * tan_q - (vec[1] + vec[2])

        delta = np.power(bp, 2) - 4 * ap * cp
        if delta <= 0:
            return np.array([np.nan, np.nan])
        phi1 = 2 * np.arctan((-bp + np.sqrt(delta)) / (2 * ap))
        if np.abs(q - self.cal_pivot_q(vec, phi1)) > 1e-6:
            phi1 = np.nan
        phi2 = 2 * np.arctan((-bp - np.sqrt(delta)) / (2 * ap))
        if np.abs(q - self.cal_pivot_q(vec, phi2)) > 1e-6:
            phi2 = np.nan
        return np.array([phi1, phi2])

    def cal_hinge_phi(self, vec, q):
        delta = np.power(vec[0], 2) + np.power(vec[1], 2) - np.power(vec[2] - np.cos(q), 2)
        if delta <= 0.0:
            return np.array([np.nan, np.nan])
        phi1 = 2 * np.arctan((vec[0] + np.sqrt(delta)) / (np.cos(q) + vec[1] - vec[2]))
        if np.abs(q - self.cal_hinge_q(vec, phi1)) > 1e-6:
            phi1 = np.nan
        phi2 = 2 * np.arctan((vec[0] - np.sqrt(delta)) / (np.cos(q) + vec[1] - vec[2]))
        if np.abs(q - self.cal_hinge_q(vec, phi2)) > 1e-6:
            phi2 = np.nan
        return np.array([phi1, phi2])

    def ikine_with_avoidance(self, Teq: SE3):
        self.cal_phi_limit(Teq)

        phi_min_max_all = np.vstack((self._min_phi, self._max_phi))

        for i in range(len(self._pivot_joint_index)):
            phi_min_max = phi_min_max_all[:, self._pivot_joint_index[i]]
            ind = np.argsort(phi_min_max)
            phi_limit = np.array([-np.pi, np.pi])

            for j in range(4):
                if np.isnan(phi_min_max[ind[j]]):
                    break
                if phi_min_max[ind[j]] > self._phi:
                    phi_limit[1] = phi_min_max[ind[j]]
                else:
                    phi_limit[0] = phi_min_max[ind[j]]
            self._phi_limit_all[:, self._pivot_joint_index[i]] = phi_limit

        for i in range(len(self._hinge_joint_index)):
            phi_min_max = phi_min_max_all[:, self._hinge_joint_index[i]]
            ind = np.argsort(phi_min_max)
            phi_limit = np.array([-np.pi, np.pi])

            for j in range(4):
                if np.isnan(phi_min_max[ind[j]]):
                    break
                if phi_min_max[ind[j]] > self._phi:
                    phi_limit[1] = phi_min_max[ind[j]]
                else:
                    phi_limit[0] = phi_min_max[ind[j]]
            self._phi_limit_all[:, self._hinge_joint_index[i]] = phi_limit

        self._phi_limit = np.array([np.max(self._phi_limit_all[0, :]), np.min(self._phi_limit_all[1, :])])

        new_phi = self._phi + self._K * ((self._phi_limit[1] - self._phi_limit[0]) / 2) * (
                np.exp(-self._alpha * (
                        (self._phi - self._phi_limit[0]) / (self._phi_limit[1] - self._phi_limit[0]))) - np.exp(
            -self._alpha * ((self._phi_limit[1] - self._phi) / (self._phi_limit[1] - self._phi_limit[0]))))

        self._phi = self._filter_factor * self._phi + (1.0 - self._filter_factor) * new_phi

        qs = np.zeros(self._dof)
        qs[3] = self.q0[3]
        for i in range(len(self._hinge_joint_index)):
            qs[self._hinge_joint_index[i]] = self.cal_hinge_q(self._hinge_vecs[:, i], self._phi)

        for i in range(len(self._pivot_joint_index)):
            qs[self._pivot_joint_index[i]] = self.cal_pivot_q(self._pivot_vecs[:, i], self._phi)
        q0_s = list(map(wrap, self.q0))

        for i in range(self._dof):
            if qs[i] - q0_s[i][0] > np.pi:
                qs[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif qs[i] - q0_s[i][0] < -np.pi:
                qs[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                qs[i] += q0_s[i][1] * 2 * np.pi

        return qs
