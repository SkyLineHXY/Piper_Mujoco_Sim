import numpy as np
from manipulator_grasp.arm.constanst import MathConst


class MathUtils:
    @staticmethod
    def near_zero(value: float) -> bool:
        # return np.abs(value) < MathConst.EPS
        return np.abs(value) < 1e-6
