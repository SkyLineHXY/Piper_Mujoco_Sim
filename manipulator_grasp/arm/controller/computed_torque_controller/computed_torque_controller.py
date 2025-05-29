import numpy as np

from ..controller import Controller
from ..pid_controller import PIDController
from arm.robot import Robot


class ComputedTorqueController(Controller):

    def __init__(self, kps: list, kis: list, kds: list, robot: Robot, ts=0.001, filter_coefficient=100.0) -> None:
        self._robot = robot
        self._pid_controllers = [PIDController(kps[i], kis[i], kds[i], ts, filter_coefficient) for i in
                                 range(robot.dof)]
        self._qd_prev = np.zeros(robot.dof)
        self._dqd_prev = np.zeros(robot.dof)
        self._ts = ts

    def control(self, qd, q):
        dqd: np.ndarray = (qd - self._qd_prev) / self._ts
        ddqd: np.ndarray = (dqd - self._dqd_prev) / self._ts

        pid_outs = [self._pid_controllers[i].control(qd[i], q[i]) for i in range(self._robot.dof)]

        self._qd_prev = np.array(qd)
        self._dqd_prev = np.array(dqd)

        return self._robot.inv_dynamics(qd, dqd, ddqd + pid_outs)

    def set_qd(self, qd: np.ndarray):
        self._qd_prev = qd.copy()
