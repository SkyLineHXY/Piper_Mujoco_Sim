import os.path
import sys

# import mujoco_py
import spatialmath as sm
import time
from multipledispatch import dispatch
# from mujoco_py import load_model_from_path, MjSim
from manipulator_grasp.arm.geometry import SE3Impl
import numpy as np
import casadi
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import mujoco
import mujoco.viewer
from scipy.optimize import least_squares
from manipulator_grasp.arm.robot import Robot
import pinocchio as pin
# from pinocchio import casadi as cpin
from transformations import quaternion_from_euler, euler_from_quaternion

from pinocchio import casadi as cpin
import threading
ROOT_DIR =os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
# from manipulator_grasp.arm.motion_planning import *
# from manipulator_grasp.utils import mj
# ---------------
# ---------------
# 逆运动学求解器（使用 Pinocchio）
# ------------------------------
parent_dir = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/Kinova7DoF-MuJoCo'
class IKSolver:
    def __init__(self, urdf_path: str):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        # 获取末端执行器的 Frame ID（假设 URDF 中定义了名为 'grasp_link' 的框架）
        self.frame_id = self.model.getFrameId("grasp_link")
        # 关节上下限，用于设置优化约束
        self.lower = self.model.lowerPositionLimit
        self.upper = self.model.upperPositionLimit
        # 创建 CasADi 版的 Pinocchio 模型和数据，用于符号化计算
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

    def solve(self, qpos, target_pose:pin.SE3):
        """
        求解逆运动学：
        - qpos: 当前关节角向量 (长度 7 或 8)
        - target_pose: 目标末端姿态，pinocchio.SE3 类型
        返回优化后的 8 维关节角向量 joint_ctrl。
        """
        # 转换为 numpy 数组
        qpos = np.array(qpos).flatten()
        qpos = np.array(qpos).flatten()
        nq=8
        if qpos.size == nq:
            q_init = qpos.copy()
        elif qpos.size == nq - 1:
            q_init = np.zeros(nq)
            q_init[:nq - 1] = qpos
            q_init[-1] = q_init[-2]*-1  # 假设第8维为 prismatic 关节，初始为0
        else:
            raise ValueError(f"qpos 维度应为 {nq} 或 {nq - 1}，但收到 {qpos.size}")

        # 构造目标位姿的 CasADi SE3 常量
        R_casadi = casadi.SX(target_pose.rotation)
        p_casadi = casadi.SX(target_pose.translation)
        target_casadi = cpin.SE3(R_casadi, p_casadi)
        # 创建 CasADi 优化变量
        opti = casadi.Opti()
        # 使用 SX 符号变量替代 MX 变量
        q_sx = casadi.SX.sym('q', nq)  # 改为 SX 类型变量
        cpin.framesForwardKinematics(self.cmodel, self.cdata, q_sx)
        # 计算末端当前位姿到目标位姿的相对变换，并取 6D 对数误差
        err = cpin.log6(self.cdata.oMf[self.frame_id].inverse() * target_casadi).vector
        reg_coeff = 1e-3
        cost_sx = casadi.sumsqr(err) + reg_coeff * casadi.sumsqr(q_sx - q_init)
        cost_func = casadi.Function("cost_func", [q_sx], [cost_sx])
        q_var = opti.variable(nq)
        cost = cost_func(q_var)
        opti.minimize(cost)

        # 关节上下限约束
        for i in range(nq):
            opti.subject_to(q_var[i] >= self.lower[i])
            opti.subject_to(q_var[i] <= self.upper[i])
        # 初始值
        opti.set_initial(q_var, q_init)
        # 选择求解器
        opti.solver("ipopt")
        # 尝试求解
        try:
            sol = opti.solve()
            if sol.stats()["success"]:
                q_sol = sol.value(q_var)
            else:
                print("IK 求解失败，返回初始关节角")
                q_sol = q_init
        except RuntimeError as e:
            # 求解失败时，输出警告并返回初值
            print("IK 求解失败，返回初始关节角：", e)
            # print("当前变量值：", opti.debug.value(q_var))
            q_sol = q_init
        # 确保返回的是一维数组
        joint_ctrl = np.array(q_sol).flatten()
        return joint_ctrl
    def forward_kinematics(self, q):

        if len(q) == 7 and isinstance(q,list):
            q.append(q[-1]*-1)
            q = np.asarray(q, dtype=np.float64)
        elif isinstance(q,np.ndarray):
            if len(q) == 7:
                q = q.tolist()
                q.append(q[-1]*-1)
                q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.frame_id]
class IKOptimizer:
    def __init__(self, urdf_path: str, ee_frame_name: str = 'grasp_link'):
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path,package_dirs=[parent_dir])
        self.model = self.robot.model
        self.collision_model = self.robot.collision_model

        # self.collision_model.addAllCollisionPairs()
        for i in range(4, 9):
            for j in range(0, 3):
                self.collision_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.collision_model)
        self.visual_model = self.robot.visual_model
        self.ee_frame_name = ee_frame_name
        self.ee_frame_id =  self.model.getFrameId(self.ee_frame_name)
        self.data = self.model.createData()

        self.target_translation = np.array([0.155, 0.0, 0.222])  # ✅ 更新初始末端位置



        q = quaternion_from_euler(0, 0, 0)
        self.model.addFrame(
            pin.Frame('ee',
                      self.model.getJointId('joint6'),
                      pin.SE3(
                          # pin.Quaternion(1, 0, 0, 0),
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([0.0, 0.0, 0.1]),
                      ),
                      pin.FrameType.OP_FRAME)
        )#末端姿态
        self.gripper_id = self.model.getFrameId("ee")
        self.q_current = np.zeros(8)  # 初始关节角度
        self.q_current[0]=0.5
        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)

        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector,
                )
            ],
        )
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.model.nq)
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit)
        )
        self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization)
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)
    def objective(self, q, target_pose):
        current_pose = self.forward_kinematics(q)
        error = pin.log(current_pose.inverse() * target_pose)
        return error

    def forward_kinematics(self, q):

        if len(q) == 7 and isinstance(q,list):
            q.append(q[-1]*-1)
            q = np.asarray(q, dtype=np.float64)
        elif isinstance(q,np.ndarray):
            if len(q) == 7:
                q = q.tolist()
                q.append(q[-1]*-1)
                q = np.asarray(q, dtype=np.float64)
        pin.forwardKinematics(self.robot.model, self.data, q)
        pin.updateFramePlacements(self.robot.model, self.data)
        return self.data.oMf[self.ee_frame_id]
    def ik_fun_agelix(self,target_pose,q0):
        if len(q0) == 7 and isinstance(q0,np.ndarray):
            q0 = q0.tolist()
            q0.append(q0[-1]*-1)
            q0 = np.asarray(q0, dtype=np.float64)

        self.init_data = q0

        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.param_tf, casadi.DM(target_pose.homogeneous))
        try:
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)
            max_diff = max(abs(self.q_current - sol_q))
            if max_diff > 15.0 / 180.0 * 3.1415:
                # print("Excessive changes in joint angle:", max_diff)
                self.init_data = np.zeros(8)
            # print("max_diff:", max_diff)
            # q = sol_q
            # if max_diff > 30.0/180.0*3.1415:
            #     # print("Excessive changes in joint angle:", max_diff)
            #     self.init_data = np.zeros(8)
            # else:
            #     self.init_data = sol_q
            # self.q_current = sol_q
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            return sol_q, '', False
        # self.q_current = sol_q
        return sol_q

    def solve(self, target_pose, q0):
        if len(q0) == 7 and isinstance(q0,np.ndarray):
            q0 = q0.tolist()
            q0.append(q0[-1]*-1)
            q0 = np.asarray(q0, dtype=np.float64)

        lb = np.full_like(q0, -3.14)
        ub = np.full_like(q0, 3.14)
        res = least_squares(self.objective, q0, args=(target_pose,), bounds=(lb, ub))
        return res.x

class PiperGraspEnv:
    def __init__(self):
        # 初始化 MuJoCo 模型和数据
        # filename = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene_piper.xml')

        # mj_filename = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/Kinova7DoF-MuJoCo/piper_description/mujoco_model/scene_piper.xml'
        # urdf_path = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/Kinova7DoF-MuJoCo/piper_description/urdf/piper_description.urdf'
        mj_filename = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/scenes/scene_grasp.xml'
        urdf_path = '/home/zzq/Desktop/jichuan/YOLO_World-SAM-GraspNet/manipulator_grasp/assets/agilex_piper/piper_description/urdf/piper_description.urdf'
        # self.ik_solver = IKOptimizer(urdf_path, ee_frame_name='grasp_link')
        self.ik_solver = IKSolver(urdf_path)
        self.mj_model = mujoco.MjModel.from_xml_path(mj_filename)
        self.mj_data = mujoco.MjData(self.mj_model)

        # self.sim = MjSim(self.mj_model)
        # self.arm_control = ArmControl(self.sim)

        self.height = 640 # 256 640 720
        self.width = 640 # 256 640 1280
        # 创建两个渲染器实例，分别用于生成彩色图像和深度图
        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        # self.sim_hz = 500
        self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        # 为了方便观察
        self.mj_viewer.cam.lookat[:] = [1.8, 1.1, 1.7]  # 对应XML中的center
        self.mj_viewer.cam.azimuth = 210      # 对应XML中的azimuth
        self.mj_viewer.cam.elevation = -35    # 对应XML中的elevation
        self.mj_viewer.cam.distance = 1.2     # 根据场景调整的距离值
        self.mj_viewer.sync() # 立即同步更新


    def reset(self):
        self.target_translation = np.array([0.19, 0.0, 0.3])  # ✅ 更新初始末端位置
        self.target_rotation = np.array([  # ✅ 更新初始末端旋转矩阵
            [-0.64925909, -0.03433515, 0.7597919],
            [-0.05494111, 0.99848793, -0.001826],
            [-0.7585, -0.0429297, -0.65016378]
        ])
        self.q_current = np.array([0,1.128,-1.327,0,1.636,0,0])# 初始关节角度
        # self.mj_data.qpos[:8] = self.q_current
        # self.q_current = np.zeros(8)  # 初始关节角度
        # 更新渲染器中的场景数据
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        # 启用深度渲染
        self.mj_depth_renderer.enable_depth_rendering()
        # self.step(action=self.q_current)
    def render_Thread(self):
        while True:
            self.step()
    def render(self):
        '''
        常用于强化学习或机器人控制任务中，提供环境的视觉观测数据。
        '''
        # 更新渲染器中的场景数据
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        # 渲染图像和深度图
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }

    def step(self, action=None):
        if action is not None:

            self.mj_data.ctrl = action

        mujoco.mj_step(self.mj_model, self.mj_data)
        self.mj_viewer.sync()

    # @overload
    # def move_to_position_no_planner(self, target_pose:SE3Impl)-> None:
    #     target_translation = target_pose.A[:3][-1]
    #     target_rotation = target_pose.A[:3][:3]
    #     target_pose = pin.SE3(target_rotation, target_translation)
    #     q_new = self.ik_solver.solve(target_pose, self.q_current)
    #     self.q_current = q_new
    #     self.step(self.q_current)
    #
    # @overload
    # def move_to_position_no_planner(self, x:float, y:float, z:float, rx:float, ry:float, rz:float):
    #     # 平移更新
    #     self.target_translation = np.array([x, y, z])
    #
    #     # 旋转矩阵更新
    #     rot_x = R.from_euler('x', rx, degrees=False).as_matrix()
    #     rot_y = R.from_euler('y', ry, degrees=False).as_matrix()
    #     rot_z = R.from_euler('z', rz, degrees=False).as_matrix()
    #     self.target_rotation = rot_x @ rot_y @ rot_z
    #     target_pose = pin.SE3(self.target_rotation, self.target_translation)
    #     q_new = self.ik_solver.solve(target_pose, self.q_current)
    #     self.q_current = q_new
    #     self.step(self.q_current)

    def move_to_position_no_planner(self, *args):
        if len(args) == 1 and isinstance(args[0], np.ndarray):
            target_pose_orin = args[0]
            target_translation = target_pose_orin[:3,-1]
            target_rotation = target_pose_orin[:3,:3]
        elif len(args) == 6:
            x, y, z, rx, ry, rz = args
            target_translation = np.array([x, y, z])
            rot_x = R.from_euler('x', rx, degrees=False).as_matrix()
            rot_y = R.from_euler('y', ry, degrees=False).as_matrix()
            rot_z = R.from_euler('z', rz, degrees=False).as_matrix()
            target_rotation = rot_x @ rot_y @ rot_z

        else:
            raise TypeError("Invalid arguments for move_to_position_no_planner")

        target_pose = pin.SE3(target_rotation, target_translation)
        q_new = self.ik_solver.solve(self.q_current,target_pose)[:6]
        # q_new = self.ik_solver.solve(target_pose, self.q_current)[:6]

        self.q_current[:6] = q_new
        self.step(self.q_current)
    def set_piper_qpos(self,q0):
        self.q_current = q0
    # def move_to_position(self, x, y, z, rx, ry, rz):
    #     # 平移更新
    #     self.target_translation = np.array([x, y, z])
    #
    #     # 旋转矩阵更新
    #     rot_x = R.from_euler('x', rx, degrees=False).as_matrix()
    #     rot_y = R.from_euler('y', ry, degrees=False).as_matrix()
    #     rot_z = R.from_euler('z', rz, degrees=False).as_matrix()
    #     self.target_rotation = rot_x @ rot_y @ rot_z
    #     target_pose = pin.SE3(self.target_rotation, self.target_translation)
    #     q_new = self.ik_solver.solve(target_pose, self.q_current)
    #     print(f"Planning a path...")
    #     q_path = self.ik_solver.planner.plan(self.q_current, q_new)
    #     if q_path is not None:
    #         if len(q_path) > 0:
    #             print(f"Got a path with {len(q_path)} waypoints")
    #
    #             def plot_joint_path(q_path):
    #                 """
    #                 绘制RRT planner生成的离散关节轨迹点
    #
    #                 参数
    #                 ----
    #                 q_path : list of array-like
    #                     每个元素是一个关节位置 array，size=8
    #                 """
    #                 q_path = np.array(q_path)  # 转成numpy数组，shape = (N, 8)
    #                 num_points, num_joints = q_path.shape
    #                 plt.figure(figsize=(10, 6))
    #                 for joint_idx in range(num_joints):
    #                     plt.plot(range(num_points), q_path[:, joint_idx], label=f'Joint {joint_idx + 1}')
    #
    #                 plt.xlabel('Waypoint Index')
    #                 plt.ylabel('Joint Position (rad)')
    #                 plt.title('RRT Planned Joint Trajectory')
    #                 plt.legend()
    #                 plt.grid(True)
    #                 plt.tight_layout()
    #                 plt.show()
    #                 data = self.ik_solver.model.createData()
    #                 ee_positions = []
    #                 for q in q_path:
    #                     ee_pose = self.ik_solver.forward_kinematics(q)
    #                     ee_positions.append(copy.deepcopy(ee_pose.translation))
    #                 ee_positions = np.array(ee_positions)  # shape = (N, 3)
    #                 # 末端轨迹3D绘图
    #                 fig = plt.figure(figsize=(8, 6))
    #                 ax = fig.add_subplot(111, projection='3d')
    #                 ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], marker='o')
    #
    #                 ax.set_xlabel('X (m)')
    #                 ax.set_ylabel('Y (m)')
    #                 ax.set_zlabel('Z (m)')
    #                 ax.set_title('End-Effector Trajectory in Cartesian Space')
    #                 ax.grid(True)
    #                 plt.tight_layout()
    #                 plt.show()
    #             # plot_joint_path(q_path)
    #             Optim_options = CubicTrajectoryOptimizationOptions(
    #                 num_waypoints=len(q_path),
    #                 samples_per_segment=1,
    #                 min_segment_time=0.5,
    #                 max_segment_time=10.0,
    #                 # min_vel=-1.5,
    #                 # max_vel=1.5,
    #                 # min_accel=-0.75,
    #                 # max_accel=0.75,
    #                 # min_jerk=-1.0,
    #                 # max_jerk=1.0,
    #                 max_planning_time=1.0,
    #                 check_collisions=True,
    #                 min_collision_dist=0.0,
    #                 collision_influence_dist=0.02,
    #                 collision_avoidance_cost_weight=0.0,
    #                 collision_link_list=[],
    #             )
    #             optimizer = CubicTrajectoryOptimization(self.ik_solver.model,
    #                                                     self.ik_solver.collision_model,
    #                                                     Optim_options)
    #
    #             traj = optimizer.plan(q_path, init_path=q_path)
    #             # traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)
    #             if traj is not None:#TODO
    #                 print("Trajectory optimization successful")
    #                 traj_gen = traj.generate(0.025)
    #                 self.q_vec = traj_gen[1]
    #                 print(f"path has {self.q_vec.shape[1]} points")
    #                 # 动作控制执行
    #                 for i in range(self.q_vec.shape[1]):
    #                     q_target = self.q_vec[:, i]
    #                     self.step(q_target)
    #                     time.sleep(0.025)
    #                 tforms = extract_cartesian_poses(self.ik_solver.model, "grasp_link", self.q_vec.T)
    #                 positions = []
    #                 for tform in tforms:
    #                     position = tform.translation
    #                     positions.append(position)
    #                 positions = np.array(positions)
    #                 # for position in positions:
    #
    #                 fig = plt.figure()
    #                 ax = fig.add_subplot(111, projection='3d')
    #                 # 绘制位置轨迹
    #                 ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], marker='o')
    #                 # 绘制姿态
    #                 for i, tform in enumerate(tforms):
    #                     position = tform.translation
    #                     rotation_matrix = tform.rotation
    #                     # 提取坐标轴方向的向量
    #                     x_axis = rotation_matrix[:, 0]
    #                     y_axis = rotation_matrix[:, 1]
    #                     z_axis = rotation_matrix[:, 2]
    #                     # 绘制坐标轴向量
    #                     ax.quiver(position[0], position[1], position[2],
    #                               x_axis[0], x_axis[1], x_axis[2], color='r', length=0.01)
    #                     ax.quiver(position[0], position[1], position[2],
    #                               y_axis[0], y_axis[1], y_axis[2], color='g', length=0.01)
    #                     ax.quiver(position[0], position[1], position[2],
    #                               z_axis[0], z_axis[1], z_axis[2], color='b', length=0.01)
    #                     # 设置坐标轴标签
    #                 ax.set_xlabel('X')
    #                 ax.set_ylabel('Y')
    #                 ax.set_zlabel('Z')
    #                 # 显示图形
    #                 plt.show()
    #             else:
    #                 self.q_current = q_new
    #                 self.step(self.q_current)
    #
    #         else:
    #             print("Failed to plan.")
    #             self.step()
    #     else:
    #         self.q_current = q_new
    #         self.step(q_new)

    def get_joint(self):
        """
            获取当前机械臂关节状态
        """
        return self.mj_data.qpos[:7]
    def get_end_pose(self):
        """
            获取当前末端位姿
        """
        pose = self.ik_solver.forward_kinematics(self.mj_data.qpos[:8])
        print(pose)
    def gripper_control(self,gripper_W: float):
        self.q_current[-1] = gripper_W
        # self.q_current[-1] = gripper_W
        self.step(self.q_current)
    def run_circle_trajectory(self, center, radius, angular_speed, duration):
        #TODO
        """
        让机械臂末端沿圆轨迹运动

        Args:
            center: 圆心位置 (np.array [x, y, z])
            radius: 半径 (float)
            angular_speed: 角速度 (rad/s)
            duration: 总时长 (秒)
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            t = time.time() - start_time
            angle = angular_speed * t

            # 圆轨迹计算：XY平面圆
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            # 固定旋转（或者你也可以让它同步旋转）
            euler =R.from_matrix(self.target_rotation).as_euler('xyz',degrees=False)
            rx, ry, rz = euler[0], euler[1], euler[2]
            # 调用 move_to_position 控制机械臂运动
            self.move_to_position_no_planner(x, y, z, rx, ry, rz)
            # 控制频率（和 sim_hz 配合）
            time.sleep(1.0 / 20)  # 100 Hz

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()

if __name__ == '__main__':
    env = PiperGraspEnv()
    env.reset()
    while True:
        env.step()
    # env.get_end_pose()
    # env.run_circle_trajectory([0.3,0.2,0.3],0.1,0.1,100)
    # env.run_line_trajectory([0.35, 0.3, 0.3],[0.25, 0.0, 0.3],0.05,20)
    env.close()

