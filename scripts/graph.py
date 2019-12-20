#!/usr/bin/python
import copy
import itertools
import operator
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from PIL import Image
import time
import matplotlib.animation as animation
import pickle
import os
from os import path
import cv2
import rospy
from threading import Lock
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

FREE = 0
OCCUPIED = 100
UNKNOWN = -1
SMALL = 0.000000000001
INITIAL_SIZE = 6
MIN_WIDTH = 0.5 / 0.05
# COMM_RANGE = 100 / 0.05
COMM_RANGE = 400
PRECISION = 1
EDGE_LENGTH = MIN_WIDTH  # / 4.0
SCAN_RADIUS = 5.0 / 0.05
INFO_SCAN_RADIUS = 5.0
THETA_SC = 360
PI = math.pi
ERROR_MARGIN = 0.03
RANGE_MARGIN = 1.0
INDEX_FOR_X = 1
INDEX_FOR_Y = 0
SLOPE_MARGIN = 0.1


class Graph:
    def __init__(self, robot_id=-1):
        self.call_count = 0
        self.pixel_desc = {}
        self.obstacles = {}
        self.resolution = None
        self.point = None
        self.origin_x = None
        self.origin_y = None
        self.grid_values = np.zeros((1, 1))
        self.robot_id = robot_id
        self.is_active = False
        self.lock = Lock()

        self.obstacles = {}
        self.adj_list = {}
        self.leaves = {}
        self.edges = {}

        self.old_edges = {}
        self.old_leaf_slope = {}
        self.old_obstacles = {}
        self.old_adj_list = {}

        self.leaf_slope = {}
        self.longest = None
        self.adj_dict = {}
        self.tree_size = {}
        self.leave_dict = {}
        self.parent_dict = {}

        self.new_information = {}
        self.known_points = {}
        self.unknown_points = {}

    def update_occupacygrid(self, occ_grid, pose):
        start_time = time.time()
        self.resolution = occ_grid.info.resolution
        origin_pos = occ_grid.info.origin.position
        self.origin_x = origin_pos.x
        self.origin_y = origin_pos.y
        self.grid_values = np.array(occ_grid.data).reshape((occ_grid.info.height, occ_grid.info.width)).astype(
            np.float32)
        point = self.pose2pixel(pose)
        polygon = self.create_polygon(point)
        self.get_image_desc(polygon)
        self.compute_hallway_points(polygon)
        now = time.time()
        t = now - start_time
        if self.robot_id == 0:
            rospy.logerr("Robot {}: Update map time: {}".format(self.robot_id, t))
        self.save_data([{'time': now, 'type': 0, 'robot_id': self.robot_id, 'computational_time': t}],
                       "plots/map_update_time.pickle")

    def compute_hallway_points(self, polygon):
        obstacles = list(self.obstacles)
        try:
            vor = Voronoi(obstacles)
            vertices = vor.vertices
            ridge_vertices = vor.ridge_vertices
            ridge_points = vor.ridge_points
            self.edges.clear()
            for i in range(len(ridge_vertices)):
                ridge_vertex = ridge_vertices[i]
                ridge_point = ridge_points[i]
                if ridge_vertex[0] != -1 or ridge_vertex[1] != -1:
                    p1 = [0] * 2
                    p2 = [0] * 2
                    p1[INDEX_FOR_X] = round(vertices[ridge_vertex[0]][INDEX_FOR_X], 2)
                    p1[INDEX_FOR_Y] = round(vertices[ridge_vertex[0]][INDEX_FOR_Y], 2)
                    p2[INDEX_FOR_X] = round(vertices[ridge_vertex[1]][INDEX_FOR_X], 2)
                    p2[INDEX_FOR_Y] = round(vertices[ridge_vertex[1]][INDEX_FOR_Y], 2)
                    if self.in_range(p1, polygon) or self.in_range(p2, polygon):
                        p1 = tuple(p1)
                        p2 = tuple(p2)
                        e = (p1, p2)
                        if e not in self.edges and self.is_free(p1) and self.is_free(p2):
                            q1 = obstacles[ridge_point[0]]
                            q2 = obstacles[ridge_point[1]]
                            o = (tuple(q1), tuple(q2))
                            if self.D(q1, q2) > 1 / self.resolution:  # MIN_WIDTH:
                                self.edges[e] = o
            self.get_adjacency_list(self.edges)
            self.connect_subtrees()
            self.merge_similar_edges()
        except Exception as e:
            rospy.logerr("Robot {}: Obstacle size: {}".format(self.robot_id, len(obstacles)))

    def connect_subtrees(self):
        N = len(self.adj_list)
        self.get_deeepest_tree_source()
        if not N:
            return
        if N == len(self.adj_dict[self.longest]):
            self.get_adjacency_list(self.edges)
            return
        allnodes = list(self.adj_list)
        for leaf, adj in self.adj_dict.items():
            adj_leaves = self.leave_dict[leaf]
            leaf_dist = {}
            for l in adj_leaves:
                dist_dict = {}
                for n2 in allnodes:
                    if n2 not in adj:
                        dist_dict[n2] = self.D(l, n2)
                if dist_dict:
                    nei = min(dist_dict, key=dist_dict.get)
                    leaf_dist[l] = nei
            longest_dist = {k: self.D(k, v) for k, v in leaf_dist.items()}
            if longest_dist:
                closest = min(longest_dist, key=longest_dist.get)
                close_node = leaf_dist[closest]
                cl_n = self.adj_list[close_node]
                cls_n = self.adj_list[closest]
                self.adj_list[closest].append(close_node)
                self.adj_list[close_node].append(closest)
                w1 = 0
                w2 = 0
                clw1 = None
                clw2 = None
                for cl in cl_n:
                    clw1 = (close_node, cl)
                    if clw1 in self.edges:
                        w1 = self.edges[(close_node, cl)]
                        break
                for ad in cls_n:
                    clw2 = (closest, ad)
                    if (closest, ad) in self.edges:
                        w2 = self.edges[(closest, ad)]
                        break
                if w1 > w2:
                    self.edges[(close_node, closest)] = clw1
                else:
                    self.edges[(close_node, closest)] = clw2
        self.get_adjacency_list(self.edges)
        return self.connect_subtrees()

    def get_adjacency_list(self, edge_dict):
        edge_list = list(edge_dict)
        self.adj_list.clear()
        self.leaf_slope.clear()
        for e in edge_list:
            first = e[0]
            second = e[1]
            if first in self.adj_list:
                self.adj_list[first].append(second)
            else:
                self.adj_list[first] = [second]
            if second in self.adj_list:
                self.adj_list[second].append(first)
            else:
                self.adj_list[second] = [first]
        for k, v in self.adj_list.items():
            if len(v) == 1:
                self.leaf_slope[k] = self.theta(v[0], k)

    def get_deeepest_tree_source(self):
        self.leave_dict.clear()
        self.adj_dict.clear()
        self.tree_size.clear()
        self.parent_dict.clear()
        leaves = list(self.leaf_slope)
        for s in leaves:
            S = [s]
            visited = []
            parents = {s: None}
            lf = []
            node_adjlist = {}
            while len(S) > 0:
                u = S.pop()
                if u in self.adj_list:
                    neighbors = self.adj_list[u]
                    if u not in node_adjlist:
                        node_adjlist[u] = neighbors
                    if len(neighbors) == 1:
                        lf.append(u)
                    for v in neighbors:
                        if v not in visited:
                            S.append(v)
                            parents[v] = u
                    visited.append(u)
            self.adj_dict[s] = node_adjlist
            self.tree_size[s] = len(visited)
            self.leave_dict[s] = lf
            self.parent_dict[s] = parents
        if self.tree_size:
            self.longest = max(self.tree_size, key=self.tree_size.get)

    def merge_similar_edges(self):
        if not self.adj_list:
            return
        self.get_deeepest_tree_source()
        tree_leaves = self.leave_dict[self.longest]
        parents = self.parent_dict[self.longest]
        new_edges = {}
        visited = {}
        for m in tree_leaves:
            leaf = copy.deepcopy(m)
            u = parents[leaf]
            while u is not None:
                # us = self.slope(u, leaf)
                us = self.get_vector(leaf, u)
                if parents[u] is None:
                    self.add_edge((u, leaf), new_edges)
                    break
                else:
                    if u not in visited:
                        # ps = self.slope(parents[u], u)
                        ps = self.get_vector(u, parents[u])
                        # if abs(us + ps) > SLOPE_MARGIN:
                        cos_theta, separation = self.compute_similarity(us, ps)
                        match = 0.9 <= cos_theta <= 1
                        if not match:
                            if parents[u] is not None:
                                if (u, leaf) not in new_edges:
                                    self.add_edge((u, leaf), new_edges)
                                    leaf = u
                                pars = copy.deepcopy(parents)
                                for k, p in pars.items():
                                    if p == leaf and k not in visited:
                                        parents[k] = u
                        visited[u] = None
                        u = parents[u]
                    else:
                        self.add_edge((u, leaf), new_edges)
                        break
        self.get_adjacency_list(new_edges)
        self.edges = new_edges

        # backup
        self.old_edges = copy.deepcopy(self.edges)
        self.old_leaf_slope = copy.deepcopy(self.leaf_slope)
        self.old_adj_list = copy.deepcopy(self.adj_list)

    def get_frontiers(self, pose, count):
        self.lock.acquire()
        frontiers = []
        start_time = time.time()
        self.compute_new_information()
        ppoints = []
        while len(self.new_information) > 0:
            best_point = max(self.new_information, key=self.new_information.get)
            new_p = self.pixel2pose(best_point)
            frontiers.append(new_p)
            ppoints.append(best_point)
            if len(frontiers) == count:
                break
            del self.new_information[best_point]
        now = time.time()
        t = (now - start_time)
        if self.robot_id == 0:
            rospy.logerr("Robot {}: Get frontiers: {}".format(self.robot_id, time.time() - start_time))
        self.save_data([{'time': now, 'type': 2, 'robot_id': self.robot_id, 'computational_time': t}],
                       "plots/map_update_time.pickle")
        if self.robot_id == 0:
            self.plot_data(None, pose, ppoints, is_initial=True)
        self.lock.release()
        return frontiers

    def compute_intersections(self, robot_pose):
        self.lock.acquire()
        start = time.time()
        intersecs = []
        close_edge = []
        robot_pose = self.pose2pixel(robot_pose)
        vertex_dict = {}
        ri_dict = {}
        closest_ridge = {}
        edge_list = list(self.old_edges)
        for e in edge_list:
            p1 = e[0]
            p2 = e[1]
            o = self.old_edges[e]
            width = self.D(o[0], o[1])
            if self.D(p1, robot_pose) < COMM_RANGE and self.D(p1, p2) > 1 / self.resolution:
                u_check = self.W(p1, robot_pose) < width  # or self.W(p2, robot_pose) < width
                if u_check:
                    v = self.get_vector(p1, p2)
                    desc = (v, width)
                    vertex_dict[e] = desc
                    distance1 = self.D(p1, robot_pose)
                    closest_ridge[distance1] = p1
                    ri_dict[p1] = e
        if closest_ridge:
            close_point = closest_ridge[min(closest_ridge.keys())]
            close_edge = ri_dict[close_point]
            intersecs = self.process_decision(vertex_dict, close_edge)
            now = time.time()
            t = (now - start)
            if self.robot_id == 0:
                rospy.logerr("Robot {}: Compute intersections time: {}".format(self.robot_id, t))
            self.save_data([{'time': now, 'type': 1, 'robot_id': self.robot_id, 'computational_time': t}],
                           "plots/map_update_time.pickle")
            self.plot_intersections(None, close_edge, intersecs, robot_pose)
        self.lock.release()
        return close_edge, intersecs

    def process_decision(self, vertex_descriptions, ridge):
        r_desc = vertex_descriptions[ridge]
        v1 = r_desc[0]
        w1 = r_desc[1]
        p1 = ridge[0]
        p2 = ridge[1]
        intersections = []
        for j, desc in vertex_descriptions.items():
            if j != ridge:
                p3 = j[1]
                v2 = desc[0]
                w2 = desc[1]
                cos_theta, separation = self.compute_similarity(v1, v2)
                if self.D(p2, p3) > SCAN_RADIUS:
                    # width_diff = abs(w1 - w2)
                    if -1 <= cos_theta <= -0.9:
                        # if width_diff < 1 / self.resolution:
                        if separation <= 0.1 / self.resolution:
                            if self.collinear(p1, p2, p3, w1):
                                # if self.there_is_unknown_region(p2, p3):
                                intersections.append((ridge, j))
                                # break
                                # else:
                                #     rospy.logerr("Is in unknown: {}".format(self.there_is_unknown_region(p2, p3)))
                                #     pass
                            else:
                                rospy.logerr("Collinear: {}".format(self.collinear(p1, p2, p3, w1)))
                                pass
                        else:
                            # rospy.logerr("Separation: {}".format(separation))
                            pass
                    else:
                        # rospy.logerr("Cos theta: {}".format(cos_theta))
                        pass
        return intersections

    def there_is_unknown_region(self, p1, p2):
        x_min = int(round(min([p1[1], p2[1]])))
        y_min = int(round(min([p1[0], p2[0]])))
        x_max = int(round(max([p1[1], p2[1]])))
        y_max = int(round(max([p1[0], p2[0]])))
        points = []
        point_count = 0
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                point_count += 1
                region_point = [0] * 2
                region_point[INDEX_FOR_X] = x
                region_point[INDEX_FOR_Y] = y
                region_point = tuple(region_point)
                if region_point in self.pixel_desc and self.pixel_desc[region_point] == UNKNOWN:
                    points.append(region_point)
        return len(points) >= point_count / 2.0

    def compute_new_information(self):
        resolution = 1
        frontier_size = {}
        self.new_information.clear()
        self.known_points.clear()
        self.unknown_points.clear()
        for leaf, slope in self.old_leaf_slope.items():
            if leaf not in self.obstacles:
                obs = None
                if (self.old_adj_list[leaf][0], leaf) in self.old_edges:
                    obs = self.old_edges[(self.old_adj_list[leaf][0], leaf)]
                elif (leaf, self.old_adj_list[leaf][0]) in self.old_edges:
                    obs = self.old_edges[(leaf, self.old_adj_list[leaf][0])]
                if obs:
                    frontier_size[leaf] = self.D(obs[0], obs[1])
                    ks, us = self.area(leaf, slope)
                    unknown_area = len(us) * resolution ** 2
                    if leaf not in self.new_information:
                        self.new_information[leaf] = unknown_area
                    self.known_points[leaf] = ks
                    self.unknown_points[leaf] = us

    def next_states(self, row, col, scale=10):
        neighbors = []
        for i in range(scale):
            values = []
            right = [0] * 2
            down = [0] * 2
            up = [0] * 2
            left = [0] * 2
            right[INDEX_FOR_X] = row + i
            right[INDEX_FOR_Y] = col
            right = tuple(right)

            down[INDEX_FOR_X] = row
            down[INDEX_FOR_Y] = col - 1
            down = tuple(down)

            up[INDEX_FOR_X] = row
            up[INDEX_FOR_Y] = col + 1
            up = tuple(up)

            left[INDEX_FOR_X] = row - i
            left[INDEX_FOR_Y] = col
            left = tuple(left)

            if right in self.pixel_desc:
                values.append(self.pixel_desc[right])
            if down in self.pixel_desc:
                values.append(self.pixel_desc[down])
            if up in self.pixel_desc:
                values.append(self.pixel_desc[up])
            if left in self.pixel_desc:
                values.append(self.pixel_desc[left])
            neighbors += values
        return neighbors

    def pose2pixel(self, pose):
        x = round((pose[INDEX_FOR_X] - self.origin_x) / self.resolution)
        y = round((pose[INDEX_FOR_Y] - self.origin_y) / self.resolution)
        position = [0] * 2
        position[INDEX_FOR_X] = x
        position[INDEX_FOR_Y] = y
        return tuple(position)

    def pixel2pose(self, point):
        new_p = [0] * 2
        new_p[INDEX_FOR_Y] = round(self.origin_x + point[INDEX_FOR_X] * self.resolution, 2)
        new_p[INDEX_FOR_X] = round(self.origin_y + point[INDEX_FOR_Y] * self.resolution, 2)
        return tuple(new_p)

    def theta(self, p, q):
        dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
        dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
        return math.atan2(dy, dx)

    def D(self, p, q):
        dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
        dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
        return math.sqrt(dx ** 2 + dy ** 2)

    def T(self, p, q):
        return self.D(p, q) * math.cos(self.theta(p, q))

    def W(self, p, q):
        return abs(self.D(p, q) * math.sin(self.theta(p, q)))

    def slope(self, p, q):
        dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
        dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
        if dx == 0:
            dx = SMALL
            if dy < 0:
                return -1 / dx
            return 1 / dx
        return dy / dx

    def get_vector(self, p1, p2):
        xv = p2[INDEX_FOR_X] - p1[INDEX_FOR_X]
        yv = p2[INDEX_FOR_Y] - p1[INDEX_FOR_Y]
        v = [0] * 2
        v[INDEX_FOR_X] = xv
        v[INDEX_FOR_Y] = yv
        v = tuple(v)
        return v

    def area(self, point, orientation):
        resolution = 1
        known_points = []
        unknown_points = []
        for d in np.arange(0, INFO_SCAN_RADIUS, resolution):
            distance_points = []
            for theta in range(-1 * THETA_SC // 2, THETA_SC + 1):
                angle = np.deg2rad(theta) + orientation
                x = point[INDEX_FOR_X] + d * np.cos(angle)
                y = point[INDEX_FOR_Y] + d * np.sin(angle)
                new_p = [0] * 2
                new_p[INDEX_FOR_X] = x
                new_p[INDEX_FOR_Y] = y
                distance_points.append(tuple(new_p))
            unique_points = list(set(distance_points))
            for p in unique_points:
                if self.is_free(p) or self.is_obstacle(p):
                    known_points.append(p)
                else:
                    unknown_points.append(p)
        return known_points, unknown_points

    def connect_root(self, node, prev_obst, new_edges):
        if len(self.adj_dict) > 1:
            dist = {}
            for leaf, adj in self.adj_dict.items():
                if node not in adj:
                    for k, l in adj.items():
                        dist[k] = self.D(k, node)
            if dist:
                nei = min(dist, key=dist.get)
                new_edges[(nei, node)] = prev_obst

    def add_edge(self, edge, new_edges):
        obst = ((0, 0), (0, 0))
        p1 = edge[0]
        neighbors = self.adj_list[p1]
        for n in neighbors:
            if (p1, n) in self.edges:
                o = self.edges[(p1, n)]
                new_edges[edge] = o
                obst = 0
                break
            if (n, p1) in self.edges:
                o = self.edges[(n, p1)]
                new_edges[edge] = o
                obst = o
                break
        return obst

    def round_point(self, p):
        return (round(p[INDEX_FOR_X], PRECISION), round(p[INDEX_FOR_Y], PRECISION))

    def compute_similarity(self, v1, v2):
        dv1 = np.sqrt(v1[INDEX_FOR_X] ** 2 + v1[INDEX_FOR_Y] ** 2)
        dv2 = np.sqrt(v2[INDEX_FOR_X] ** 2 + v2[INDEX_FOR_Y] ** 2)
        dotv1v2 = v1[INDEX_FOR_X] * v2[INDEX_FOR_X] + v1[INDEX_FOR_Y] * v2[INDEX_FOR_Y]
        v1v2 = dv1 * dv2
        if v1v2 == 0:
            v1v2 = 1
        if abs(dotv1v2) == 0:
            dotv1v2 = 0
        cos_theta = round(dotv1v2 / v1v2, 3)
        separation = round(dv1 * math.sin(math.acos(cos_theta)), 2)
        return cos_theta, separation

    def get_linear_points(self, intersections):
        linear_ridges = []
        for intersect in intersections:
            p1 = intersect[0][0]
            p2 = intersect[0][1]
            p3 = intersect[1][1]
            if self.D(p2, p3) > SCAN_RADIUS and self.collinear(p1, p2, p3):
                linear_ridges.append(intersect)
        return linear_ridges

    def collinear(self, p1, p2, p3, width):
        s1 = self.slope(p1, p2)
        s2 = self.slope(p2, p3)
        if s1 == s2 or self.W(p2, p3) < width:
            return True
        return False

    def there_is_unknown_region(self, p1, p2):
        x_min = int(round(min([p1[1], p2[1]])))
        y_min = int(round(min([p1[0], p2[0]])))
        x_max = int(round(max([p1[1], p2[1]])))
        y_max = int(round(max([p1[0], p2[0]])))
        points = []
        point_count = 0
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                point_count += 1
                region_point = [0] * 2
                region_point[INDEX_FOR_X] = x
                region_point[INDEX_FOR_Y] = y
                region_point = tuple(region_point)
                if region_point in self.pixel_desc and self.pixel_desc[region_point] == UNKNOWN:
                    points.append(region_point)
        return len(points) >= point_count / 2.0

    def get_directed_adjacency_list(self, hallway_vertices):
        adj_list = {}
        for e in hallway_vertices:
            first = e[0]
            second = e[1]
            if first in adj_list:
                adj_list[first].append(second)
            else:
                adj_list[first] = [second]
        nodes = [k for k, v in adj_list.items() if len(v) == 1]
        return adj_list, nodes

    def process_edges(self):
        x_pairs = []
        y_pairs = []
        edge_list = list(self.old_edges)
        for edge in edge_list:
            xh, yh = self.reject_outliers(list(edge))
            if len(xh) == 2:
                x_pairs.append(xh)
                y_pairs.append(yh)
        return x_pairs, y_pairs

    def is_free(self, p):
        xc = np.round(p[INDEX_FOR_X])
        yc = np.round(p[INDEX_FOR_Y])
        new_p = [0] * 2
        new_p[INDEX_FOR_X] = xc
        new_p[INDEX_FOR_Y] = yc
        new_p = tuple(new_p)
        return new_p in self.pixel_desc and self.pixel_desc[new_p] == FREE  # self.free_points

    def is_obstacle(self, p):
        xc = np.round(p[INDEX_FOR_X])
        yc = np.round(p[INDEX_FOR_Y])
        new_p = [0] * 2
        new_p[INDEX_FOR_X] = xc
        new_p[INDEX_FOR_Y] = yc
        new_p = tuple(new_p)
        return new_p in self.pixel_desc and self.pixel_desc[new_p] == OCCUPIED

    def reject_outliers(self, data):
        raw_x = [v[INDEX_FOR_X] for v in data]
        raw_y = [v[INDEX_FOR_Y] for v in data]
        rejected_points = [v for v in raw_x if v < 0]
        indexes = [i for i in range(len(raw_x)) if raw_x[i] in rejected_points]
        x_values = [raw_x[i] for i in range(len(raw_x)) if i not in indexes]
        y_values = [raw_y[i] for i in range(len(raw_y)) if i not in indexes]
        return x_values, y_values

    def plot_intersections(self, ax, ridge, intersections, point):
        obstacles = list(self.old_obstacles)
        # unknowns = list(self.unknowns)
        if not ax:
            fig, ax = plt.subplots(figsize=(16, 10))

        # ux = [v[INDEX_FOR_X] for v in unknowns]
        # uy = [v[INDEX_FOR_Y] for v in unknowns]

        x = [v[INDEX_FOR_X] for v in obstacles]
        y = [v[INDEX_FOR_Y] for v in obstacles]
        current_x = [ridge[0][INDEX_FOR_X], ridge[1][INDEX_FOR_X]]
        current_y = [ridge[0][INDEX_FOR_Y], ridge[1][INDEX_FOR_Y]]
        target_x = []
        target_y = []

        for section in intersections:
            Q = section[1]
            target_x += [Q[0][INDEX_FOR_X], Q[1][INDEX_FOR_X]]
            target_y += [Q[0][INDEX_FOR_Y], Q[1][INDEX_FOR_Y]]
        for i in range(len(target_x)):
            ax.plot(target_x[i], target_y[i], "g-d")

        for i in range(len(current_x)):
            ax.plot(current_x[i], current_y[i], "r-d")

        ax.scatter(point[INDEX_FOR_X], point[INDEX_FOR_Y], marker='*', color='purple')
        ax.scatter(x, y, color='black', marker='s')
        # ax.scatter(ux, uy, color='gray', marker='1')

        x_pairs, y_pairs = self.process_edges()
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "m-.")
        plt.grid()
        plt.savefig("plots/intersections_{}_{}.png".format(self.robot_id, time.time()))  # TODO consistent time.
        plt.close()
        # plt.show()

    def plot_data(self, ax, fp, frontiers, is_initial=False, vertext=None):
        # unknown_points = list(self.unknown_points)
        obstacles = list(self.old_obstacles)
        # known_points = list(self.known_points)
        leaves = list(self.old_leaf_slope)
        if not ax:
            fig, ax = plt.subplots(figsize=(16, 10))

        # ux = [v[INDEX_FOR_X] for v in unknown_points]
        # uy = [v[INDEX_FOR_Y] for v in unknown_points]

        xr = [v[INDEX_FOR_X] for v in obstacles]
        yr = [v[INDEX_FOR_Y] for v in obstacles]
        ax.scatter(xr, yr, color='black', marker="s")
        x_pairs, y_pairs = self.process_edges()
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "g-.")
        ax.scatter(fp[INDEX_FOR_X], fp[INDEX_FOR_Y], color='blue', marker='s')  # Pose in x, y -- col, row.

        if vertext:
            ax.scatter(vertext[INDEX_FOR_X], vertext[INDEX_FOR_Y], color='purple', marker=">")
        for leaf in leaves:
            ax.scatter(leaf[INDEX_FOR_X], leaf[INDEX_FOR_Y], color='red', marker='*')
        # ax.scatter(ux, uy, color='gray', marker='1')
        if is_initial:
            for leaf in leaves:
                if leaf in self.known_points and leaf in self.unknown_points:
                    ks = self.known_points[leaf]
                    us = self.unknown_points[leaf]
                    kx = [p[INDEX_FOR_X] for p in ks]
                    ky = [p[INDEX_FOR_Y] for p in ks]
                    ux = [p[INDEX_FOR_X] for p in us]
                    uy = [p[INDEX_FOR_Y] for p in us]
                    ax.scatter(ux, uy, color='purple', marker="3")
                    ax.scatter(kx, ky, color='gray', marker="3")
                    ax.scatter(leaf[INDEX_FOR_X], leaf[INDEX_FOR_Y], color='red', marker='*')

            fx = [p[INDEX_FOR_X] for p in frontiers]
            fy = [p[INDEX_FOR_Y] for p in frontiers]
            ax.scatter(fx, fy, color='goldenrod', marker="P")
        plt.grid()
        plt.savefig("plots/plot_{}_{}.png".format(self.robot_id, time.time()))
        plt.close()
        # plt.show()

    def get_image_desc(self, polygon):
        self.lock.acquire()
        height = self.grid_values.shape[0]
        width = self.grid_values.shape[1]
        self.obstacles.clear()
        for row in range(height):
            for col in range(width):
                index = [0] * 2
                index[INDEX_FOR_X] = col
                index[INDEX_FOR_Y] = row
                if self.in_range(index, polygon):
                    index = tuple(index)
                    p = self.grid_values[row, col]
                    if p == -1:
                        self.pixel_desc[index] = UNKNOWN
                    elif -1 < p <= 0:
                        self.pixel_desc[index] = FREE
                    elif p > 0:
                        self.pixel_desc[index] = OCCUPIED
                        self.obstacles[index] = None
        self.old_obstacles = copy.deepcopy(self.obstacles)
        self.lock.release()

    def in_range(self, point, polygon):
        x = point[INDEX_FOR_X]
        y = point[INDEX_FOR_Y]
        return polygon[0][0] <= x <= polygon[2][0] and polygon[0][1] <= y <= polygon[2][1]

    def create_polygon(self, pose):
        x = pose[INDEX_FOR_X]
        y = pose[INDEX_FOR_Y]
        first = (x - COMM_RANGE, y - COMM_RANGE)
        second = (x - COMM_RANGE, y + COMM_RANGE)
        third = (x + COMM_RANGE, y + COMM_RANGE)
        fourth = (x + COMM_RANGE, y - COMM_RANGE)
        ranges = [first, second, third, fourth]
        return ranges

    def save_data(self, data, file_name):
        saved_data = []
        if not path.exists(file_name):
            f = open(file_name, "wb+")
            f.close()
        else:
            saved_data = self.load_data_from_file(file_name)
        saved_data += data
        with open(file_name, 'wb') as fp:
            pickle.dump(saved_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            fp.close()

    def load_data_from_file(self, file_name):
        data_dict = []
        if path.exists(file_name) and path.getsize(file_name) > 0:
            with open(file_name, 'rb') as fp:
                try:
                    data_dict = pickle.load(fp)
                except Exception as e:
                    rospy.logerr("error: {}".format(e))
        return data_dict


if __name__ == "__main__":
    graph = Graph()
    filename = "map_message1.pickle"
    # graph.load_map_data(filename)
