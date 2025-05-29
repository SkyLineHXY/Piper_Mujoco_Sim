from typing import Tuple
import numpy as np
from spatialmath import SE3

from ..simplex import Point, Line, LineSegment, Support
from ..shape import Plane, Brick
from .GJK import GJK


class Distance:
    @staticmethod
    def point_to_point(point0: Point, point1: Point):
        return np.linalg.norm((point1 - point0).get_t())

    @staticmethod
    def point_to_plane(point: Point, plane: Plane):
        return np.dot(point.get_t() - plane.get_t(), plane.get_normal_vector())

    @staticmethod
    def point_to_line(point: Point, line: Line) -> float:
        foot_point, t = Distance.__calculate_foot_point(point, line)
        return np.linalg.norm(point.get_t() - foot_point)

    @staticmethod
    def point_to_line_segment(point: Point, line_segment: LineSegment) -> float:
        foot_point, t = Distance.__calculate_foot_point(point, line_segment)

        t = max(0, min(1, t))

        projection = line_segment.get_point0().get_t() + t * (
                line_segment.get_point1().get_t() - line_segment.get_point0().get_t())
        return np.linalg.norm(point.get_t() - projection)

    @staticmethod
    def point_to_brick(point: Point, brick: Brick):
        p = brick.base.inv() * SE3.Trans(*point.get_t()).t
        p = p.squeeze()

        if p[2] < -brick.dimensions[2] / 2.0:
            if p[1] < -brick.dimensions[1] / 2.0:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_point(point, brick.xn_yn_zn_point)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_point(point, brick.xp_yn_zn_point)
                else:
                    return Distance.point_to_line_segment(point, brick.yn_zn_line_segment)
            elif p[1] > brick.dimensions[1] / 2.0:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_point(point, brick.xp_yn_zn_point)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_point(point, brick.xp_yp_zn_point)
                else:
                    return Distance.point_to_line_segment(point, brick.yp_zn_line_segment)
            else:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_line_segment(point, brick.xn_zn_line_segment)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_line_segment(point, brick.xp_zn_line_segment)
                else:
                    return Distance.point_to_plane(point, brick.zn_plane)
        elif p[2] > brick.dimensions[2] / 2.0:
            if p[1] < -brick.dimensions[1] / 2.0:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_point(point, brick.xn_yn_zp_point)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_point(point, brick.xp_yn_zp_point)
                else:
                    return Distance.point_to_line_segment(point, brick.yn_zp_line_segment)
            elif p[1] > brick.dimensions[1] / 2.0:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_point(point, brick.xp_yn_zp_point)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_point(point, brick.xp_yp_zp_point)
                else:
                    return Distance.point_to_line_segment(point, brick.yp_zp_line_segment)
            else:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_line_segment(point, brick.xn_zp_line_segment)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_line_segment(point, brick.xp_zp_line_segment)
                else:
                    return Distance.point_to_plane(point, brick.zp_plane)
        else:
            if p[1] < -brick.dimensions[1] / 2.0:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_line_segment(point, brick.xn_yn_line_segment)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_line_segment(point, brick.xp_yn_line_segment)
                else:
                    return Distance.point_to_plane(point, brick.yn_plane)
                    pass
            elif p[1] > brick.dimensions[1] / 2.0:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_line_segment(point, brick.xn_yp_line_segment)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_line_segment(point, brick.xp_yp_line_segment)
                else:
                    return Distance.point_to_plane(point, brick.yp_plane)
            else:
                if p[0] < -brick.dimensions[0] / 2.0:
                    return Distance.point_to_plane(point, brick.xn_plane)
                if p[0] > brick.dimensions[0] / 2.0:
                    return Distance.point_to_plane(point, brick.xp_plane)
                else:
                    return -1.0
        return -1.0

    @staticmethod
    def line_segment_to_line_segment(line_segment0: LineSegment, line_segment1: LineSegment):
        a = np.power(np.linalg.norm(line_segment0.get_point1().get_t() - line_segment0.get_point0().get_t()), 2)
        b = np.dot(line_segment0.get_point1().get_t() - line_segment0.get_point0().get_t(),
                   line_segment1.get_point1().get_t() - line_segment1.get_point0().get_t())
        c = np.power(np.linalg.norm(line_segment1.get_point1().get_t() - line_segment1.get_point0().get_t()), 2)
        d = np.dot(line_segment0.get_point1().get_t() - line_segment0.get_point0().get_t(),
                   line_segment0.get_point0().get_t() - line_segment1.get_point0().get_t())
        e = np.dot(line_segment1.get_point1().get_t() - line_segment1.get_point0().get_t(),
                   line_segment0.get_point0().get_t() - line_segment1.get_point0().get_t())
        f = np.power(np.linalg.norm(line_segment0.get_point0().get_t() - line_segment1.get_point0().get_t()), 2)

        det = a * c - np.power(b, 2)
        if det > 0.0:
            bte = b * e
            ctd = c * d
            if bte <= ctd:
                if e <= 0.0:
                    s = 1.0 if -d >= a else (-d / a if -d > 0.0 else 0.0)
                    t = 0.0
                elif e < c:
                    s = 0.0
                    t = e / c
                else:
                    s = 1.0 if b - d >= a else ((b - d) / a if b - d > 0.0 else 0.0)
                    t = 1.0
            else:
                s = bte - ctd
                if s >= det:
                    if b + e <= 0.0:
                        s = 0.0 if -d <= 0.0 else (-d / a if -d < a else 1.0)
                        t = 1.0
                    elif b + e < c:
                        s = 1.0
                        t = (b + e) / c
                    else:
                        s = (0.0 if b - d <= 0 else ((b - d) / a if b - d < a else 1.0))
                        t = 1.0
                else:
                    ate = a * e
                    btd = b * d
                    if ate < btd:
                        s = 0.0 if -d <= 0.0 else (1.0 if -d >= a else -d / a)
                        t = 0.0
                    else:
                        t = ate - btd
                        if t >= det:
                            s = 0.0 if b - d <= 0.0 else (1.0 if b - d >= a else (b - d) / a)
                            t = 1.0
                        else:
                            s /= det
                            t /= det
        else:
            if e <= 0.0:
                s = 0.0 if -d <= 0.0 else (1.0 if -d >= a else -d / a)
                t = 0.0
            elif e >= c:
                s = 0.0 if b - d <= 0.0 else (1.0 if b - d >= a else (b - d) / a)
                t = 1.0
            else:
                s = 0.0
                t = e / c

        distance = np.sqrt(a * np.power(s, 2) - 2 * b * s * t + c * np.power(t, 2) + 2 * d * s - 2 * e * t + f)

        return distance

    @staticmethod
    def __calculate_foot_point(point: Point, line: Line) -> tuple:
        v = line.get_point0().get_t()
        w = line.get_point1().get_t()
        p = point.get_t()

        if np.array_equal(v, w):
            return np.linalg.norm(p - v), 0
        l2 = (w - v).dot(w - v)

        t = (p - v).dot(w - v) / l2
        foot_point = v + t * (w - v)
        return foot_point, t

    @staticmethod
    def calculate_distance(shape0: Support, shape1: Support) -> float:
        return GJK.calculate_distance(shape0, shape1)

    @staticmethod
    def calculate_distance_and_points(shape0: Support, shape1: Support) -> Tuple[float, Tuple]:
        return GJK.calculate_distance_and_points(shape0, shape1)
