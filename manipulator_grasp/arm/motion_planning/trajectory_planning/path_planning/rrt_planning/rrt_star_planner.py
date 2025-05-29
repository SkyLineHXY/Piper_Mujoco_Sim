from typing import List
import multiprocessing
import numpy as np

from arm.geometry import LineSegment, Distance

from .node import Node
from .rrt_map import RRTMap
from .rrt_parameter import RRTParameter
from .rrt_planner import RRTPlanner


class RRTStarPlanner(RRTPlanner):
    def __init__(self, rrt_map: RRTMap, rrt_parameter: RRTParameter, pool: multiprocessing.Pool = None) -> None:
        self._radius = rrt_parameter.radius
        super().__init__(rrt_map, rrt_parameter, pool)

    def find_near_nodes(self, new_node: Node) -> List[int]:
        n_node = len(self.nodes)
        r = self._radius * self._expand_dis * np.sqrt((np.log(n_node) / n_node))
        d_list = [Distance.point_to_point(node.get_point(), new_node.get_point()) for node in self.nodes]
        near_inds = [d_list.index(d) for d in d_list if d <= r]
        return near_inds

    def choose_parent(self, new_node: Node, near_inds: List[int], pool: multiprocessing.Pool = None) -> Node:
        if len(near_inds) == 0:
            return new_node

        d_list = []
        for i in near_inds:
            if not self._check_collision.check_collision(LineSegment(self.nodes[i].get_point(), new_node.get_point()),
                                                         pool):
                d = Distance.point_to_point(self.nodes[i].get_point(), new_node.get_point())
                d_list.append(self.nodes[i].cost + d)
            else:
                d_list.append(float("inf"))

        min_cost = min(d_list)
        min_ind = near_inds[d_list.index(min_cost)]

        if min_cost == float('inf'):
            return new_node

        new_node.set_cost(min_cost)
        new_node.set_parent(min_ind)

        return new_node

    def rewire(self, new_node: Node, near_inds: List[int], pool: multiprocessing.Pool = None) -> None:
        n_node = len(self.nodes)
        for i in near_inds:
            near_node = self.nodes[i]

            d = Distance.point_to_point(near_node.get_point(), new_node.get_point())

            s_cost = new_node.cost + d

            if near_node.cost > s_cost:
                if not self._check_collision.check_collision(LineSegment(near_node.get_point(), new_node.get_point()),
                                                             pool):
                    near_node.parent = n_node - 1
                    near_node.cost = s_cost

    def add_node(self, node: Node, pool: multiprocessing.Pool = None) -> None:
        near_inds = self.find_near_nodes(node)
        node = self.choose_parent(node, near_inds, pool)

        super().add_node(node)
        self.rewire(node, near_inds, pool)

    def get_path_and_length(self) -> bool:
        if self._path_length > self.get_path_length():
            super().get_path_and_length()
            self.update_parameter()

        return False

    def set_goal(self, pool: multiprocessing.Pool = None) -> None:
        super().set_goal(pool)

        near_inds = self.find_near_nodes(self.goal)
        self.goal = self.choose_parent(self.goal, near_inds, pool)

    def update_parameter(self):
        pass
