import numpy as np

from ..controller import Controller
from arm.robot import Robot


class AdaptiveController(Controller):
    def __init__(self, kds: list, robot: Robot, ts=0.001, filter_coefficient=100.0) -> None:
        super().__init__()
        self._robot = robot
        parameters = self._robot.inertial_parameters
        parameters[0:30] *= 0.8
        self._robot.inertial_parameters = parameters
        self._qd_prev = np.zeros(robot.dof)
        self._dqd_prev = np.zeros(robot.dof)
        self._q_prev = np.zeros(robot.dof)
        self._kds = np.array(kds)
        self._ts = ts
        self._Fai = 5.0 * np.ones(self._robot.dof)
        self._Tau = np.zeros(self._robot.inertial_parameters.size)
        self._Tau[:30] = 0.2

    def control(self, qd, q):
        dqd: np.ndarray = (qd - self._qd_prev) / self._ts
        ddqd: np.ndarray = (dqd - self._dqd_prev) / self._ts
        dq: np.ndarray = (q - self._q_prev) / self._ts

        dqr = dqd + self._Fai * (qd - q)
        ddqr = ddqd + self._Fai * (dqd - dq)

        tau = self._robot.inv_dynamics_adaptive(q, dq, dqr, ddqr) + self._kds * (dqr - dq)

        self._update(q, dq, dqr, ddqr)
        self._qd_prev = np.array(qd)
        self._dqd_prev = np.array(dqd)
        self._q_prev = np.array(q)

        return tau

    def _update(self, q, dq, dqr, ddqr):
        Y = self._robot.get_adaptive_identification_matrix(q, dq, dqr, ddqr)
        r = dqr - dq
        dp = self._Tau * (Y.T @ r)
        self._robot.inertial_parameters += dp * self._ts
