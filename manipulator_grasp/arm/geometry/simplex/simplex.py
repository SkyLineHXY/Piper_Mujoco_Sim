from abc import ABC, abstractmethod
from typing import List
from .geometry import Geometry


class Simplex(ABC):

    @abstractmethod
    def calculate_closest_point_to_origin(self) -> Geometry:
        pass

    def calculate_barycentric_coordinates(self, geometry: Geometry) -> List[float]:
        pass

