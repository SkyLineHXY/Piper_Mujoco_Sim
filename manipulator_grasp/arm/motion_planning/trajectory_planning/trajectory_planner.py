from .path_planning import PathPlanner
from .velocity_planning import VelocityPlanner

from .trajectory_parameter import TrajectoryParameter


class TrajectoryPlanner:

    def __init__(self, parameter: TrajectoryParameter) -> None:
        super().__init__()

        self.path_planner = PathPlanner(parameter.get_path_parameter())
        self.velocity_planner = VelocityPlanner(parameter.get_velocity_parameter())

    def interpolate(self, t: float):
        s: float = self.velocity_planner.interpolate(t)
        return self.path_planner.interpolate(s)
