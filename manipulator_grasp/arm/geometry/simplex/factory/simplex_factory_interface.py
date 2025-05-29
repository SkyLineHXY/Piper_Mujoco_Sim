from .simplex_parameter import SimplexParameter
from manipulator_grasp.arm.geometry.simplex.simplex import Simplex


class SimplexFactoryInterface:
    def create_product(self, simplex_parameter: SimplexParameter) -> Simplex:
        pass
