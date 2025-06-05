import numpy as np

from manipulator_grasp.arm.interface import StrategyWrapper


class PositionPlanner(StrategyWrapper):

    def interpolate(self, s) -> np.ndarray:
        return self.strategy.interpolate(s)
