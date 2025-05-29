import math
import random
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from spatialmath import SO3, SE3

from arm.geometry import UnitVector, LineSegment, Circle2D, Distance2D, Collision2D
from .node import Node
from ..cartesian_planning import CartesianParameter, CartesianPlanner, LinePositionParameter, OneAttitudeParameter

from .rrt_map import RRTMap
from .rrt_parameter import RRTParameter


class RRT2DPlanner:
    def __init__(self, rrt_map: RRTMap, rrt_parameter: RRTParameter):
        self.start: Node = Node(rrt_parameter.start)
        self.goal: Node = Node(rrt_parameter.goal)
        self.area = rrt_map.area
        self.expand_dis = rrt_parameter.expand_dis
        self.goal_sample_rate = rrt_parameter.goal_sample_rate
        self.max_iter = rrt_parameter.max_iter
        self.obstacles = [Circle2D(obstacle[:-1], obstacle[-1]) for obstacle in rrt_map.obstacles]
        self.collision = Collision2D(self.obstacles)
        self.nodes = None
        self.path = None
        self.path_len = 0.0
        self.plan(rrt_parameter.animation)

    def plan(self, animation):
        start_time = time.time()

        self.nodes = [self.start]
        path = None

        for i in range(self.max_iter):
            rnd = self.sample()
            n_ind = self.get_nearest_list_index(rnd)

            nearest_node = self.nodes[n_ind]
            new_node = self.get_new_node(n_ind, rnd)

            new_line_segment = LineSegment(nearest_node.get_point(), new_node.get_point())
            collision = self.collision.check_line_segment(new_line_segment)

            if not collision:
                self.nodes.append(new_node)
                if animation:
                    self.draw_graph(new_node, path)

                if self.is_near_goal():
                    new_line_segment2 = LineSegment(new_node.get_point(), self.goal.get_point())
                    collision2 = self.collision.check_line_segment(new_line_segment2)
                    if not collision2:
                        path = self.get_final_course()
                        self.path_len = self.get_path_len()
                        print("current path lengths: {}, It costs {} s".format(self.path_len, time.time() - start_time))

                        if animation:
                            self.draw_graph(new_node, path)
                        self.path = path
                        return

    def sample(self) -> Node:
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.area[0][0], self.area[1][0]), random.uniform(self.area[0][1], self.area[1][1])]
        else:
            rnd = self.goal.get_point()
        return Node(rnd)

    def get_nearest_list_index(self, rnd: Node) -> int:
        d_list = [Distance2D.point_to_point(node.point, rnd.point) for node in self.nodes]
        min_index = d_list.index(min(d_list))
        return min_index

    def get_new_node(self, n_ind: int, rnd: Node) -> Node:
        nearest_node = self.nodes[n_ind]
        unit_vector = UnitVector(nearest_node.get_point(), rnd.get_point())
        new_point = nearest_node.get_point() + self.expand_dis * unit_vector
        new_node = Node(new_point, cost=nearest_node.cost + self.expand_dis, parent=n_ind)
        return new_node

    def draw_graph(self, rnd: Node, path):
        plt.clf()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(*rnd.get_t(), "^k")

        for node in self.nodes:
            if node.parent != -1:
                plt.plot(*zip(self.nodes[node.parent].get_t(), node.get_t()), '-g')

        for obstacle in self.obstacles:
            size = obstacle.get_radius()
            plt.plot(*obstacle.get_center().get_t(), "ok", ms=25 * size)

        plt.plot(*self.start.get_t(), 'xr')
        plt.plot(*self.goal.get_t(), 'xr')

        plt.axis((self.area[0][0], self.area[1][0], self.area[0][1], self.area[1][1]))
        plt.grid(True)
        plt.pause(0.01)

        if path is not None:
            plt.plot(*zip(*[pi.get_t().tolist() for pi in path]), '-r')
            plt.show()

    def is_near_goal(self):
        d = Distance2D.point_to_point(self.nodes[-1].get_point(), self.goal.get_point())
        if d < self.expand_dis:
            self.goal.set_cost(d + self.nodes[-1].get_cost())
            self.goal.set_parent(len(self.nodes) - 1)
            return True
        return False

    def get_final_course(self) -> List[Node]:
        path = [self.goal]
        last_index = len(self.nodes) - 1
        while self.nodes[last_index].parent != -1:
            node = self.nodes[last_index]
            path.append(node)
            last_index = node.parent
        path.append(self.start)
        return path

    def get_path_len(self) -> float:
        return self.goal.get_cost()

    def get_path_parameters(self, R0: SO3 = SO3()) -> List[CartesianParameter]:
        cartesian_parameters = []
        one_attitude_parameter = OneAttitudeParameter(R0, R0)
        points = self.path[::-1]
        for i, point in enumerate(points[:-1]):
            line_position_parameter = LinePositionParameter(point.get_t_3d(), points[i + 1].get_t_3d())
            cartesian_parameters.append(CartesianParameter(line_position_parameter, one_attitude_parameter))
        return cartesian_parameters

    def interpolate(self, s: float, R0: SO3 = SO3()):
        cartesian_parameters = self.get_path_parameters(R0)
        total_distance = s * self.path_len
        index = round(total_distance // self.expand_dis)
        local_distance = total_distance % self.expand_dis / np.linalg.norm(
            cartesian_parameters[index].get_position_parameter().get_t1() - cartesian_parameters[
                index].get_position_parameter().get_t0())
        cartesian_planner = CartesianPlanner(cartesian_parameters[index])
        return cartesian_planner.interpolate(local_distance)
