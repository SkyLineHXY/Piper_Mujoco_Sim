import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from arm.robot import Robot


def get_diff_matrices(n, order, p_init, p_end):
    K = np.zeros((n + order, n))
    fd = np.array([-1, 1, *np.zeros(n - 2)])
    T = np.diag([1])
    sign = -1

    for i in range(n - order):
        K[i + order, :] = np.roll(fd, i, 0)
    K[:order, :order] = T
    K[n: n + order, n - order: n] = -T.T

    e = np.zeros(n + order)
    return K, e


def time_obj_func(x, h):
    f = 0
    for i in range(x.size - 1):
        f += 1 / (np.sqrt(x[i]) + np.sqrt(x[i + 1]))
    return 2 * h * f


class TimeOptimalPlanner:

    def __init__(self, qs: np.ndarray, dqs: np.ndarray, ddqs: np.ndarray, robot: Robot, phi: np.ndarray,
                 alpha: np.ndarray, mu: np.ndarray, ts: float) -> None:
        super().__init__()
        self._interpolation = None
        self._robot = robot
        self._ts = ts
        self._phi = phi
        self._alpha = alpha
        self._mu = mu
        self._b_init = 0.0
        self._b_end = 0.0
        self._tf = 0.0

        self.plan(qs, dqs, ddqs)

    def plan(self, q_s: np.ndarray, dq_s: np.ndarray, ddq_s: np.ndarray):
        num_sample = 101
        h = 1 / (num_sample - 1)

        n, dim = q_s.shape
        tau = np.zeros((n - 1, dim))
        t_sample = np.zeros(n)

        b_optim = np.zeros(n)
        b_optim[0] = 0.0
        b_optim[-1] = 0.0

        # Set linear inequality constraints
        # dynamic
        D_d = np.zeros((dim * (n - 1), n - 1))
        D_c = np.zeros((dim * (n - 1), n - 1))
        g = np.zeros(dim * (n - 1))

        # kinematic
        D_dgamma = np.zeros((dim * (n - 1), n - 1))
        D_ddgamma = np.zeros((dim * (n - 1), n - 1))

        Alpha = np.kron(np.ones(n - 1), self._alpha)
        Mu = np.kron(np.ones(n - 1), self._mu)

        Lb = np.ones(n - 2) * 1e-5
        Ub = np.zeros(n - 2)

        K, e = get_diff_matrices(n - 2, 1, self._b_init, self._b_end)

        for i in range(n - 1):
            D_d[i * dim: (i + 1) * dim, i] = self._robot.get_inertia(q_s[i, :]) @ dq_s[i, :]
            D_c[i * dim: (i + 1) * dim, i] = self._robot.get_inertia(q_s[i, :]) @ ddq_s[i, :] \
                                             + self._robot.get_coriolis(q_s[i, :], dq_s[i, :]) @ dq_s[i, :]
            g[i * dim: (i + 1) * dim] = self._robot.get_gravity(q_s[i, :])

            D_dgamma[i * dim: (i + 1) * dim, i] = dq_s[i, :]
            D_ddgamma[i * dim: (i + 1) * dim, i] = ddq_s[i, :]

            if i > 0:
                Ub[i - 1] = 1 / np.max(np.power(dq_s[i, :], 2) / np.power(self._phi, 2))

        A1_t = D_d @ K / (2 * h) + D_c[:, 1:]
        C1_t = D_d @ e / (2 * h) + g + self._b_init * D_c[:, 0]

        A2_t = D_dgamma @ K / (2 * h) + D_ddgamma[:, 1:]
        C2_t = D_dgamma @ e / (2 * h) + self._b_end * D_ddgamma[:, 0]

        A = np.vstack((A1_t, -A1_t, A2_t, -A2_t))
        b = np.hstack((Mu - C1_t, Mu + C1_t, Alpha - C2_t, Alpha + C2_t))

        # solve convex problem
        constraints = ({
            'type': 'ineq',
            'fun': lambda x: b - A @ x,  # Ax - b <= 0
        })
        bounds = list(zip(Lb, Ub))
        res = minimize(time_obj_func, 0.5 * Ub, args=(h,), constraints=constraints, bounds=bounds)

        # compute outputs
        b_optim[1:-1] = res.x
        a_optim = np.diff(b_optim) / (2 * h)
        v_optim = np.sqrt(b_optim)
        dq_t = (dq_s.T * v_optim).T
        ddq_t = (dq_s[:-1, :].T * a_optim + ddq_s[:-1].T * b_optim[:-1]).T

        for i in range(n - 1):
            t_sample[i + 1] = t_sample[i] + 1 / (v_optim[i] + v_optim[i + 1])
            tau[i, :] = D_d[i * dim: (i + 1) * dim, i] * a_optim[i] + D_c[i * dim: (i + 1) * dim, i] * b_optim[i] \
                        + g[i * dim: (i + 1) * dim]
        t_sample *= 2 * h

        t_sample = np.append(t_sample, np.ceil(t_sample[-1] / self._ts) * self._ts)
        q_s = np.vstack((q_s, q_s[-1, :]))
        self._tf = t_sample[-1]
        self._interpolation = CubicSpline(t_sample, q_s)

    @property
    def tf(self) -> float:
        return self._tf

    def interpolate(self, t: float):
        if t < 0:
            t = 0
        elif t > self._tf:
            t = self._tf
        return self._interpolation(t)
