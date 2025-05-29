from typing import List

import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import modern_robotics as mr

from manipulator_grasp.arm.geometry import Geometry3D, Capsule
from manipulator_grasp.arm.utils import MathUtils
from .robot_config import RobotConfig
from .robot import Robot, get_transformation_mdh, wrap


class UR5e(Robot):

    def __init__(self) -> None:
        """
        初始化 UR5e 机械臂的几何参数、动力学参数和运动学模型。
        """
        super().__init__()
        
        # 定义机械臂的几何参数
        d1 = 0.163  # 第一关节到第二关节的偏移量
        d4 = 0.134  # 第四关节到第五关节的偏移量
        d5 = 0.1    # 第五关节到第六关节的偏移量
        d6 = 0.1    # 第六关节末端的偏移量

        a3 = 0.425  # 第三关节的长度
        a4 = 0.392  # 第四关节的长度

        self._dof = 6  # 自由度数（UR5e 有 6 个关节）
        self.q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 初始关节角度

        # DH 参数
        alpha_array = [0.0, -np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2]  # 关节轴之间的扭转角
        a_array = [0.0, 0.0, a3, a4, 0.0, 0.0]  # 关节轴之间的距离
        d_array = [d1, 0.0, 0.0, d4, d5, d6]   # 关节轴之间的偏移量
        theta_array = [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0]  # 关节初始角度
        sigma_array = [0, 0, 0, 0, 0, 0]  # 关节类型（0 表示旋转关节）

        # 定义每个关节的质量、质心位置、转动惯量和电机惯量
        m1 = 3.7  # 第一关节的质量
        r1 = np.array([0.0, 0.0, 0.0])  # 第一关节的质心位置
        I1 = np.diag([0.0102675, 0.0102675, 0.00666])  # 第一关节的转动惯量矩阵
        Jm1 = 0.1  # 第一关节的电机惯量

        m2 = 8.393  # 第二关节的质量
        r2 = np.array([0.2125, 0.0, 0.138])  # 第二关节的质心位置
        I2 = np.diag([0.0151074, 0.133886, 0.133886])  # 第二关节的转动惯量矩阵
        Jm2 = 0.1  # 第二关节的电机惯量

        m3 = 2.275  # 第三关节的质量
        r3 = np.array([0.196, 0.0, 0.007])  # 第三关节的质心位置
        I3 = np.diag([0.004095, 0.0311796, 0.0311796])  # 第三关节的转动惯量矩阵
        Jm3 = 0.1  # 第三关节的电机惯量

        m4 = 1.219  # 第四关节的质量
        r4 = np.array([0.0, 0.0, 0.0])  # 第四关节的质心位置
        I4 = np.diag([0.0025599, 0.0021942, 0.0025599])  # 第四关节的转动惯量矩阵
        Jm4 = 0.1  # 第四关节的电机惯量

        m5 = 1.219  # 第五关节的质量
        r5 = np.array([0.0, 0.0, 0.0])  # 第五关节的质心位置
        I5 = np.diag([0.0025599, 0.0025599, 0.0021942])  # 第五关节的转动惯量矩阵
        Jm5 = 0.1  # 第五关节的电机惯量

        m6 = 0.1889  # 第六关节的质量
        r6 = np.array([0.0, 0.0, -0.0228317])  # 第六关节的质心位置
        I6 = np.diag([9.90863e-05, 9.90863e-05, 0.000132134])  # 第六关节的转动惯量矩阵
        Jm6 = 0.1  # 第六关节的电机惯量

        ms = [m1, m2, m3, m4, m5, m6]  # 所有关节的质量列表
        rs = [r1, r2, r3, r4, r5, r6]  # 所有关节的质心位置列表
        Is = [I1, I2, I3, I4, I5, I6]  # 所有关节的转动惯量矩阵列表
        Jms = [Jm1, Jm2, Jm3, Jm4, Jm5, Jm6]  # 所有关节的电机惯量列表

        links = []  # 存储所有关节的链接对象
        for i in range(6):  # 遍历每个关节
            links.append(rtb.DHLink(d=d_array[i], alpha=alpha_array[i], a=a_array[i], offset=theta_array[i], mdh=True,
                                    m=ms[i], r=rs[i], I=Is[i], Jm=Jms[i], G=1.0))
        self.robot = rtb.DHRobot(links)  # 构建机械臂模型

        self.alpha_array = alpha_array  # 存储 alpha 数组
        self.a_array = a_array          # 存储 a 数组
        self.d_array = d_array          # 存储 d 数组
        self.theta_array = theta_array  # 存储 theta 数组
        self.sigma_array = sigma_array  # 存储 sigma 数组

        T = SE3()  # 初始化齐次变换矩阵
        for i in range(self.dof):  # 遍历每个关节
            Ti = get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                        self.sigma_array[i], 0.0)  # 计算每个关节的变换矩阵
            self._Ms.append(Ti.A)  # 存储变换矩阵
            T: SE3 = T * Ti  # 累积变换矩阵
            self._Ses.append(np.hstack((T.a, np.cross(T.t, T.a))))  # 存储空间向量

            Gm = np.zeros((6, 6))  # 初始化惯性矩阵
            Gm[:3, :3] = Is[i]  # 设置转动惯量部分
            Gm[3:, 3:] = ms[i] * np.eye(3)  # 设置质量部分
            AdT = mr.Adjoint(mr.RpToTrans(np.eye(3), -rs[i]))  # 计算伴随矩阵
            self._Gs.append(AdT.T @ Gm @ AdT)  # 存储惯性矩阵
            self._Jms.append(Jms[i])  # 存储电机惯量

        self._Ms.append(np.eye(4))  # 添加末尾的单位矩阵

        self.robot_config = RobotConfig()  # 初始化机器人配置对象
        self.robot_config.wrist = -1  # 设置腕部配置为默认值

    def ikine(self, Tep: SE3) -> np.ndarray:

        T06: SE3 = self._base.inv() * Tep * self._tool.inv()

        wpc = self.cal_wpc(T06)

        thetas = [0.0 for _ in range(self.dof)]

        # solve theta1
        theta1_condition = np.power(wpc[1], 2) + np.power(wpc[0], 2) - np.power(self.d_array[3], 2)
        if theta1_condition < 0:
            return np.array([])
        if MathUtils.near_zero(np.linalg.norm([wpc[1], wpc[0]])):
            thetas[0] = self.q0[0] - self.theta_array[0]  # overhead singularity
        else:
            # if self.robot_config.overhead == 0:
            #     thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3], np.sqrt(theta1_condition))
            # elif self.robot_config.overhead == 1:
            #     thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3], -np.sqrt(theta1_condition))
            # else:
            #     return np.array([])
            thetas[0] = np.arctan2(wpc[1], wpc[0]) - np.arctan2(self.d_array[3],
                                                                self.robot_config.overhead * np.sqrt(theta1_condition))

        # solve theta5
        theta5_condition = -T06.a[0] * np.sin(thetas[0]) + T06.a[1] * np.cos(thetas[0])
        if np.abs(theta5_condition) > 1:
            return np.array([])
        # if self.robot_config.wrist == 0:
        #     thetas[4] = np.arccos(theta5_condition)
        # elif self.robot_config.wrist == 1:
        #     thetas[4] = -np.arccos(theta5_condition)
        # else:
        #     return np.array([])
        thetas[4] = self.robot_config.wrist * np.arccos(theta5_condition)

        # solve theta6
        m1 = - T06.n[0] * np.sin(thetas[0]) + T06.n[1] * np.cos(thetas[0])
        n1 = - T06.o[0] * np.sin(thetas[0]) + T06.o[1] * np.cos(thetas[0])
        if MathUtils.near_zero(np.sin(thetas[4])):
            thetas[5] = self.q0[5] - self.theta_array[5]
        else:
            thetas[5] = np.arctan2(-n1 / np.sin(thetas[4]), m1 / np.sin(thetas[4]))

        # solve theta3
        T01 = get_transformation_mdh(self.alpha_array[0], self.a_array[0], self.d_array[0], thetas[0],
                                     self.sigma_array[0],
                                     0.0)
        T45 = get_transformation_mdh(self.alpha_array[4], self.a_array[4], self.d_array[4], thetas[4],
                                     self.sigma_array[4],
                                     0.0)
        T56 = get_transformation_mdh(self.alpha_array[5], self.a_array[5], self.d_array[5], thetas[5],
                                     self.sigma_array[5],
                                     0.0)
        T46 = T45 * T56
        T14 = T01.inv() * T06 * T46.inv()
        x = T14.t[0]
        y = T14.t[2]

        theta3_condition = (np.power(x, 2) + np.power(y, 2) - np.power(self.a_array[2], 2)
                            - np.power(self.a_array[3], 2)) / (2 * self.a_array[2] * self.a_array[3])
        if np.abs(theta3_condition) > 1.0:
            return np.array([])
        # if self.robot_config.inline == 0:
        #     thetas[2] = np.arccos(theta3_condition)
        # elif self.robot_config.inline == 1:
        #     thetas[2] = - np.arccos(theta3_condition)
        # else:
        #     return np.array([])
        thetas[2] = self.robot_config.inline * np.arccos(theta3_condition)

        # solve theta2
        M = np.array([
            [self.a_array[3] * np.cos(thetas[2]) + self.a_array[2], -self.a_array[3] * np.sin(thetas[2])],
            [self.a_array[3] * np.sin(thetas[2]), self.a_array[3] * np.cos(thetas[2]) + self.a_array[2]]
        ])
        XY = np.array([
            [x],
            [-y]
        ])
        CS = np.linalg.inv(M) @ XY
        thetas[1] = np.arctan2(CS[1, 0], CS[0, 0])

        # solve theta4
        thetas[3] = np.arctan2(-T14.o[0], T14.n[0]) - thetas[1] - thetas[2]

        qs = np.array([wrap(thetas[i] - self.theta_array[i])[0] for i in range(self.dof)])

        q0_s = list(map(wrap, self.q0))

        for i in range(self.dof):
            if qs[i] - q0_s[i][0] > np.pi:
                qs[i] += (q0_s[i][1] - 1) * 2 * np.pi
            elif qs[i] - q0_s[i][0] < -np.pi:
                qs[i] += (q0_s[i][1] + 1) * 2 * np.pi
            else:
                qs[i] += q0_s[i][1] * 2 * np.pi

        return qs

    def set_robot_config(self, q):
        T = self.fkine(q)
        wpc = self.cal_wpc(T)
        thetas = [q[i] + self.theta_array[i] for i in range(self.dof)]

        if wrap(np.arctan2(wpc[1], wpc[0]) - wrap(thetas[0])[0])[0] <= np.pi / 2:
            self.robot_config.overhead = 1
        else:
            self.robot_config.overhead = -1

        # inline
        if wrap(thetas[2])[0] >= 0:
            self.robot_config.inline = 1
        else:
            self.robot_config.inline = -1

        # wrist
        if wrap(thetas[4])[0] >= 0:
            self.robot_config.wrist = 1
        else:
            self.robot_config.wrist = -1

    def cal_wpc(self, T: SE3) -> np.ndarray:
        t = T.t - self.d_array[5] * T.a
        t[2] -= self.d_array[0]
        return t

    def get_geometries(self) -> List[Geometry3D]:
        Ts = []
        T = SE3()
        for i in range(self.dof):
            T = T * get_transformation_mdh(self.alpha_array[i], self.a_array[i], self.d_array[i], self.theta_array[i],
                                           self.sigma_array[i], self.q0[i])
            Ts.append(T)

        T1 = Ts[0] * SE3.Trans(0, 0, -0.04)
        geometry1 = Capsule(T1, 0.06, 0.12)

        T2 = Ts[1] * SE3.Trans(0.2, 0, 0.138) * SE3.Ry(-np.pi / 2)
        geometry2 = Capsule(T2, 0.05, 0.4)

        T3 = Ts[2] * SE3.Trans(0.2, 0, 0.007) * SE3.Ry(-np.pi / 2)
        geometry3 = Capsule(T3, 0.038, 0.38)

        T4 = Ts[3] * SE3.Trans(0.0, 0.0, -0.07)
        geometry4 = Capsule(T4, 0.04, 0.14)

        T5 = Ts[4] * SE3.Trans(0.0, 0.0, -0.06)
        geometry5 = Capsule(T5, 0.04, 0.12)

        T6 = Ts[5] * SE3.Trans(0.0, 0.0, -0.06)
        geometry6 = Capsule(T6, 0.04, 0.12)

        return [geometry1, geometry2, geometry3, geometry4, geometry5, geometry6]


if __name__ == '__main__':
    ur_robot = UR5e()
    q0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    T1 = ur_robot.fkine(q0)
    print(T1)
    ur_robot.move_cartesian(T1)
    q_new = ur_robot.get_joint()
    print(q_new)
