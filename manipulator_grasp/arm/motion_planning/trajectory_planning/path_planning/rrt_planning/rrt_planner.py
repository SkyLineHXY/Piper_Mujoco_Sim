import random
from typing import List
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from arm.geometry import UnitVector, LineSegment, Distance, Collision

from .node import Node
from .rrt_map import RRTMap
from .rrt_parameter import RRTParameter

from ..path_parameter import PathParameter
from ..joint_planning import JointParameter


class RRTPlanner:
    def __init__(self, rrt_map: RRTMap, rrt_parameter: RRTParameter, pool: multiprocessing.Pool = None) -> None:
        self._area = rrt_map.area
        self._obstacles = rrt_map.obstacles
        self.start = Node(rrt_parameter.start)
        self.goal = Node(rrt_parameter.goal)
        self._expand_dis = rrt_parameter.expand_dis
        self.goal_sample_rate = rrt_parameter.goal_sample_rate
        self.max_iter = rrt_parameter.max_iter
        self.animation = rrt_parameter.animation
        self.nodes: List[Node] = []
        self._path = []
        self._path_length = float('inf')
        self._check_collision = rrt_parameter.create_check_collision(rrt_map)
        self.plan(pool)

    def plan(self, pool: multiprocessing.Pool = None) -> None:

        self.nodes = [self.start]

        for i in range(self.max_iter):
            rnd = self.sample()
            n_ind = self.get_nearest_list_index(rnd)

            nearest_node = self.nodes[n_ind]
            new_node = self.get_new_node(n_ind, rnd)

            if self._check_collision.check_collision(LineSegment(nearest_node.get_point(), new_node.get_point()), pool):
                continue

            self.add_node(new_node, pool)

            if self.animation:
                self.draw_graph(new_node, self._path)

            if not self.is_near_goal(pool):
                continue

            if self._check_collision.check_collision(LineSegment(new_node.get_point(), self.goal.get_point()), pool):
                continue

            if self.get_path_and_length():
                break

        if self.animation:
            self.draw_graph(self.goal, self._path, True)
        return

    def sample(self) -> Node:
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(*self._area[i]) for i in range(len(self._area))]
        else:
            rnd = self.goal.get_point()
        return Node(rnd)

    def get_nearest_list_index(self, rnd: Node) -> int:
        d_list = [Distance.point_to_point(node.point, rnd.point) for node in self.nodes]
        min_index = d_list.index(min(d_list))
        return min_index

    def get_new_node(self, n_ind: int, rnd: Node) -> Node:
        nearest_node = self.nodes[n_ind]
        unit_vector = UnitVector(nearest_node.get_point(), rnd.get_point())
        new_point = nearest_node.get_point() + self._expand_dis * unit_vector
        new_node = Node(new_point, cost=nearest_node.cost + self._expand_dis, parent=n_ind)
        return new_node

    def is_near_goal(self, pool: multiprocessing.Pool = None) -> bool:
        d = Distance.point_to_point(self.nodes[-1].get_point(), self.goal.get_point())
        if d < self._expand_dis:
            self.set_goal(pool)
            return True
        return False

    def get_final_course(self) -> List[Node]:
        path = [self.goal]
        node = self.goal
        while node.parent != -1:
            node = self.nodes[node.parent]
            path.append(node)
        return path

    def get_path_length(self) -> float:
        return self.goal.get_cost()

    @property
    def success(self) -> bool:
        return self._path_length < float('inf')

    def draw_graph(self, rnd: Node, path: List, show=False) -> None:
        plt.clf()
        ax = plt.axes(projection='3d')
        if rnd is not None:
            rnd.get_point().plot(ax)

        for node in self.nodes:
            if node.parent != -1:
                LineSegment(self.nodes[node.parent].get_point(), node.get_point()).plot(ax)

        for obstacle in self._obstacles:
            obstacle.plot(ax)

        self.start.get_point().plot(ax, 'b')
        self.goal.get_point().plot(ax, 'b')

        if len(path) > 0:
            for node in path:
                if node.parent != -1:
                    LineSegment(self.nodes[node.parent].get_point(), node.get_point()).plot(ax, 'r')

        self.draw_others(ax)

        ax.set_xlim(*self._area[0])
        ax.set_ylim(*self._area[1])
        ax.set_zlim(*self._area[2])
        ax.grid(True)

        if show:
            plt.show()
        else:
            plt.pause(0.01)

    def draw_others(self, ax) -> None:
        pass

    def in_area(self, p: np.ndarray) -> bool:
        for i in range(p.size):
            if (self._area[i][0] > p[i]) or (self._area[i][1] < p[i]):
                return False
        return True

    def add_node(self, node: Node, pool: multiprocessing.Pool = None) -> None:
        self.nodes.append(node)

    def get_path_and_length(self) -> bool:
        self._path = self.get_final_course()
        self._path_length = self.get_path_length()
        return True

    def set_goal(self, pool: multiprocessing.Pool = None) -> None:
        d = Distance.point_to_point(self.nodes[-1].get_point(), self.goal.get_point())
        self.goal.set_cost(d + self.nodes[-1].get_cost())
        self.goal.set_parent(len(self.nodes) - 1)

    def get_path_parameters(self) -> List[PathParameter]:
        path_parameters = []
        points = self._path[::-1]
        for i, point in enumerate(points[:-1]):
            path_parameters.append(JointParameter(point.get_t(), points[i + 1].get_t()))
        return path_parameters
