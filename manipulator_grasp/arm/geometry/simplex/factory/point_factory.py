from .simplex_factory import SimplexFactory
from manipulator_grasp.arm.geometry.simplex.point import Point
from .simplex_parameter import SimplexParameter


class PointFactory(SimplexFactory):

    @property
    def key(self):
        return '1'

    def create_product(self, simplex_parameter: SimplexParameter):
        return Point(simplex_parameter.parameter())


point_factory = PointFactory()
point_factory.register()
