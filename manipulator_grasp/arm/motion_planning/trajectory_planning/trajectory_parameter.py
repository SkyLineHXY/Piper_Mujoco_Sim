from .path_planning import PathParameter
from .velocity_planning import VelocityParameter


class TrajectoryParameter:

    def __init__(self, path_parameter: PathParameter, velocity_parameter: VelocityParameter) -> None:
        super().__init__()

        self.path_parameter = path_parameter
        self.velocity_parameter = velocity_parameter

    def get_path_parameter(self) -> PathParameter:
        return self.path_parameter

    def get_velocity_parameter(self) -> VelocityParameter:
        return self.velocity_parameter
