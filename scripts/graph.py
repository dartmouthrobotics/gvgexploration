#!/usr/bin/python
import copy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import time
import pickle
from os import path
import rospy
from threading import Lock
from numpy.linalg import norm
from graham_scan import create_points, graham_scan

FREE = 0.0
OCCUPIED = 100.0
UNKNOWN = -1.0
SMALL = 0.000000000001
INDEX_FOR_X = 1
INDEX_FOR_Y = 0

FONT_SIZE = 16
MARKER_SIZE = 12


class Graph:
    def __init__(self, robot_id=-1):
        self.min_hallway_width = None
        self.comm_range = None
        self.point_precision = None
        self.min_edge_length = None
        self.lidar_scan_radius = None
        self.info_scan_radius = None
        self.lidar_fov = None
        self.slope_bias = None
        self.separation_bias = None
        self.opposite_vector_bias = None
        self.height = 0
        self.width = 0
        self.pixel_desc = {}
        self.obstacles = {}
        self.free_points = {}
        self.resolution = None
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

        self.computing_for_frontier = True

    def init(self, min_hallway_width, comm_range, point_precision, min_edge_length, lidar_scan_radius, info_scan_radius,
             lidar_fov, slope_bias, separation_bias, opposite_vector_bias):
        self.min_hallway_width = min_hallway_width
        self.comm_range = comm_range
        self.point_precision = point_precision
        self.min_edge_length = min_edge_length
        self.lidar_scan_radius = lidar_scan_radius
        self.info_scan_radius = info_scan_radius
        self.lidar_fov = lidar_fov
        self.slope_bias = slope_bias
        self.separation_bias = separation_bias
        self.opposite_vector_bias = opposite_vector_bias

    def update_occupacygrid(self, occ_grid, pose):
        start_time = time.time()
        self.resolution = occ_grid.info.resolution
        origin_pos = occ_grid.info.origin.position
        self.origin_x = origin_pos.x
        self.origin_y = origin_pos.y
        self.height = occ_grid.info.height
        self.width = occ_grid.info.width
        self.grid_values = np.array(occ_grid.data).reshape((self.height, self.width)).astype(
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
        obstacles = list(self.old_obstacles)
        # try:
        vor = Voronoi(obstacles)
        vertices = vor.vertices
        ridge_vertices = vor.ridge_vertices
        ridge_points = vor.ridge_points
        self.edges.clear()
        all_edges = []
        for i in range(len(ridge_vertices)):
            ridge_vertex = ridge_vertices[i]
            ridge_point = ridge_points[i]
            if ridge_vertex[0] != -1 or ridge_vertex[1] != -1:
                p1 = [0] * 2
                p2 = [0] * 2
                p1[INDEX_FOR_X] = round(vertices[ridge_vertex[0]][INDEX_FOR_X],2)
                p1[INDEX_FOR_Y] = round(vertices[ridge_vertex[0]][INDEX_FOR_Y],2)
                p2[INDEX_FOR_X] = round(vertices[ridge_vertex[1]][INDEX_FOR_X],2)
                p2[INDEX_FOR_Y] = round(vertices[ridge_vertex[1]][INDEX_FOR_Y],2)
                p1 = tuple(p1)
                p2 = tuple(p2)
                e = (p1, p2)
                all_edges.append(e)
                if e not in self.edges and self.is_free(p1) and self.is_free(p1) and self.in_range(p1,polygon) and self.in_range(p2, polygon):
                    q1 = obstacles[ridge_point[0]]
                    q2 = obstacles[ridge_point[1]]
                    o = (tuple(q1), tuple(q2))
                    if self.D(q1, q2) > self.min_hallway_width:
                        self.edges[e] = o
        self.get_adjacency_list(self.edges)
        self.connect_subtrees()
        self.merge_similar_edges()
        self.old_leaf_slope = copy.deepcopy(self.leaf_slope)
        self.old_edges = copy.deepcopy(self.edges)
        self.old_adj_list = copy.deepcopy(self.adj_list)

        # except Exception as e:
        #     rospy.logerr("Robot {}: Obstacle size: {}".format(1, len(obstacles)))

    def connect_subtrees(self):
        N = len(self.adj_list)
        self.get_deeepest_tree_source()
        if not N:
            return
        if N == len(self.adj_dict[self.longest]):
            self.get_adjacency_list(self.edges)
            return
        allnodes = []
        for k, v in self.leave_dict.items():
            allnodes += v
        allnodes = list(set(allnodes))
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
                self.adj_list[closest].add(close_node)
                self.adj_list[close_node].add(closest)
                w1 = 0
                w2 = 0
                clw1 = None
                clw2 = None
                for cl in cl_n:
                    clw1 = (close_node, cl)
                    if clw1 in self.edges:
                        cd = self.edges[(close_node, cl)]
                        w1 = self.D(cd[0], cd[1])
                        break
                for ad in cls_n:
                    clw2 = (closest, ad)
                    if (closest, ad) in self.edges:
                        cd1 = self.edges[(closest, ad)]

                        w2 = self.D(cd1[0], cd1[1])
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
                self.adj_list[first].add(second)
            else:
                self.adj_list[first] = {second}
            if second in self.adj_list:
                self.adj_list[second].add(first)
            else:
                self.adj_list[second] = {first}
        for k, v in self.adj_list.items():
            if len(v) == 1:
                self.leaf_slope[k] = self.theta(list(v)[0], k)

    def get_deeepest_tree_source(self):
        self.leave_dict.clear()
        self.adj_dict.clear()
        self.tree_size.clear()
        self.parent_dict.clear()
        leaves = list(self.leaf_slope)
        for s in leaves:
            S = [s]
            visited = {}
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
                    visited[u] = None
            self.adj_dict[s] = node_adjlist
            self.tree_size[s] = len(visited)
            self.leave_dict[s] = lf
            self.parent_dict[s] = parents
        if self.tree_size:
            self.longest = max(self.tree_size, key=self.tree_size.get)

    def merge_similar_edges(self):
        parents = {self.longest: None}
        deleted_nodes = {}
        S = [self.longest]
        visited = {}
        while len(S) > 0:
            u = S.pop()
            if u not in deleted_nodes:
                neighbors = [k for k in self.adj_list[u] if k != parents[u]]
                if len(neighbors) == 1:
                    v = neighbors[0]
                    if v not in visited:
                        S.append(v)
                        if parents[u]:
                            us = self.get_vector(parents[u], u)
                            ps = self.get_vector(u, v)
                            cos_theta, separation = self.compute_similarity(us, ps, (parents[u], u), (u, v))
                            if 1 - self.opposite_vector_bias <= cos_theta <= 1:
                                parents[v] = parents[u]
                                deleted_nodes[u] = None
                                self.adj_list[v].remove(u)
                                self.adj_list[v].add(parents[u])
                                self.adj_list[parents[u]].add(v)
                                if (u,parents[u]) in self.edges:
                                    self.edges[(parents[u],v)] = self.edges[(u,parents[u])]
                                else:
                                    self.edges[(parents[u],v)] = self.edges[(parents[u],u)]
                                del self.adj_list[u]
                            else:
                                parents[v] = u
                        else:
                            parents[v] = u
                else:
                    for v in neighbors:
                        if v not in visited:
                            S.append(v)
                            parents[v] = u
                visited[u] = None

        new_adj_list = {}
        edges = {}
        for k, v in self.adj_list.items():
            if k not in new_adj_list:
                new_adj_list[k] = []
            for l in v:
                if l not in deleted_nodes:
                    new_adj_list[k].append(l)
                    if (k,l) in self. edges:
                        edges[(k, l)] = self.edges[(k,l)]
                    else:
                        edges[(k, l)] = self.edges[(l,k)]

        for k, v in new_adj_list.items():
            if len(v) == 1:
                self.leaf_slope[k] = self.theta(k, list(v)[0])
        self.adj_list = new_adj_list
        self.edges = edges

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
            del self.new_information[best_point]
            if len(frontiers) == count:
                break

        # if len(frontiers) == count:
        now = time.time()
        t = (now - start_time)
        if self.robot_id == 0:
            rospy.logerr("Robot {}: Get frontiers: {}".format(self.robot_id, time.time() - start_time))
        self.save_data([{'time': now, 'type': 2, 'robot_id': self.robot_id, 'computational_time': t}],
                       "plots/map_update_time.pickle")
        if self.robot_id == 0:
            self.plot_data(ppoints, is_initial=True)
        # else:
        #     free_points = self.get_free_points()
        #     hull, fp = graham_scan(free_points, count, False)
        #     for p in fp:
        #         frontiers.append(self.pixel2pose(p))
        #     rospy.logerr("Hull: {}".format(frontiers))
        #     self.scatter_plot(free_points, fp, convex_hull=hull)
        self.lock.release()
        return frontiers

    def get_free_points(self):
        points = list(self.pixel_desc)
        return [p for p in points if self.pixel_desc[p] == FREE]

    def get_obstacles(self):
        points = list(self.pixel_desc)
        return [p for p in points if self.pixel_desc[p] == OCCUPIED]

    def scatter_plot(self, coords, fp, convex_hull=None):
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        pixels = list(self.pixel_desc)
        free_points = []
        obstacle_points = []
        unknown_points = []
        for p in pixels:
            v = self.pixel_desc[p]
            if v == FREE:
                free_points.append(p)
            elif v == OCCUPIED:
                obstacle_points.append(p)
            else:
                unknown_points.append(p)

        # fx, fy = zip(*free_points)
        ox, oy = zip(*obstacle_points)
        ux, uy = zip(*unknown_points)
        ax.scatter(uy, ux, marker='1', color='gray')
        # ax.scatter(fx, fy, color='white')
        ax.scatter(oy, ox, marker='1', color='black')
        xs, ys = zip(*coords)  # unzip into x and y coord lists

        # ax.scatter(xs, ys, color='purple' )  # plot the data points
        if convex_hull is not None:
            for i in range(1, len(convex_hull) + 1):
                if i == len(convex_hull): i = 0  # wrap
                c0 = convex_hull[i - 1]
                c1 = convex_hull[i]
                plt.plot((c0[INDEX_FOR_X], c1[INDEX_FOR_X]), (c0[INDEX_FOR_Y], c1[INDEX_FOR_Y]), 'm')

        x, y = zip(*fp)
        ax.plot(y, x, 'Dy')
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Y", fontsize=14)
        plt.savefig("plots/points_{}.png".format(time.time()))

    def compute_intersections(self, robot_pose):
        self.lock.acquire()
        start = time.time()
        intersecs = []
        close_edge = []
        robot_pose = self.pose2pixel(robot_pose)
        vertex_dict = {}
        closest_ridge = {}
        edge_list = list(self.old_edges)
        for e in edge_list:
            p1 = e[0]
            p2 = e[1]
            o = self.old_edges[e]
            width = self.D(o[0], o[1])
            if self.D(p1, p2) > self.min_edge_length:  # self.D(p1, robot_pose) < self.comm_range and
                u_check = self.W(p1, robot_pose) < width or self.W(p2, robot_pose) < width
                if u_check:
                    v1 = self.get_vector(p1, p2)
                    desc = (v1, width)
                    vertex_dict[e] = desc
                    d = norm(np.cross(v1, self.get_vector(robot_pose, p1))) / norm(v1)
                    closest_ridge[e] = d
        if closest_ridge:
            cr = min(closest_ridge, key=closest_ridge.get)
            if closest_ridge[cr] < vertex_dict[cr][1]:
                close_edge = cr
                intersecs = self.process_decision(vertex_dict, close_edge, robot_pose)
            now = time.time()
            t = (now - start)
            if self.robot_id == 0:
                rospy.logerr("Robot {}: Compute intersections time: {}".format(self.robot_id, t))
            self.save_data([{'time': now, 'type': 1, 'robot_id': self.robot_id, 'computational_time': t}],
                           "plots/map_update_time.pickle")
            self.plot_intersections(None, close_edge, intersecs, robot_pose)
        self.lock.release()
        return close_edge, intersecs

    def process_decision(self, vertex_descriptions, ridge, robot_pose):
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
                cos_theta, separation = self.compute_similarity(v1, v2, ridge, j)
                if self.D(robot_pose, p3) > self.lidar_scan_radius:
                    if -1 <= cos_theta <= -1 + self.opposite_vector_bias:
                        if separation <= max([w1, w2]):
                            if self.collinear(p1, p2, p3, w1):
                                if self.there_is_unknown_region(robot_pose, p3):
                                    intersections.append((ridge, j))
                                else:
                                    rospy.logerr(
                                        "Is in unknown: {}".format(self.there_is_unknown_region(robot_pose, p3)))
                            else:
                                rospy.logerr("Collinear: {}".format(self.collinear(p1, p2, p3, w1)))
                        else:
                            # rospy.logerr("Separation: {}".format(separation))
                            pass
                    else:
                        # rospy.logerr("Cos theta: {}".format(cos_theta))
                        pass
        return intersections

    def there_is_unknown_region(self, p1, p2):
        x_min = int(round(min([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])))
        y_min = int(round(min([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])))
        x_max = int(round(max([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])))
        y_max = int(round(max([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])))
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
                if (list(self.old_adj_list[leaf])[0], leaf) in self.old_edges:
                    obs = self.old_edges[(list(self.old_adj_list[leaf])[0], leaf)]
                elif (leaf, list(self.old_adj_list[leaf])[0]) in self.old_edges:
                    obs = self.old_edges[(leaf, list(self.old_adj_list[leaf])[0])]
                if obs:
                    frontier_size[leaf] = self.D(obs[0], obs[1])
                    ks, us = self.area(leaf, slope)
                    unknown_area = len(us) * resolution ** 2
                    if leaf not in self.new_information:
                        self.new_information[leaf] = unknown_area
                    self.known_points[leaf] = ks
                    self.unknown_points[leaf] = us

    def pose2pixel(self, pose):
        x = round((pose[INDEX_FOR_X] - self.origin_y) / self.resolution)
        y = round((pose[INDEX_FOR_Y] - self.origin_x) / self.resolution)
        position = [0] * 2
        position[INDEX_FOR_X] = x
        position[INDEX_FOR_Y] = y
        return tuple(position)

    def pixel2pose(self, point):
        new_p = [0] * 2
        new_p[INDEX_FOR_Y] = round(self.origin_x + point[INDEX_FOR_X] * self.resolution, self.point_precision)
        new_p[INDEX_FOR_X] = round(self.origin_y + point[INDEX_FOR_Y] * self.resolution, self.point_precision)
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
        known_points = []
        unknown_points = []
        for d in np.arange(0, self.info_scan_radius, 1):
            distance_points = []
            for theta in range(-1 * self.lidar_fov // 2, self.lidar_fov + 1):
                angle = np.deg2rad(theta) + orientation
                x = point[INDEX_FOR_X] + d * np.cos(angle)
                y = point[INDEX_FOR_Y] + d * np.sin(angle)
                new_p = [0] * 2
                new_p[INDEX_FOR_X] = x
                new_p[INDEX_FOR_Y] = y
                new_p = tuple(new_p)
                distance_points.append(new_p)
            unique_points = list(set(distance_points))
            for p in unique_points:
                if self.is_free(p) or self.is_obstacle(p):
                    known_points.append(p)
                else:
                    unknown_points.append(p)
        return known_points, unknown_points

    def compute_similarity(self, v1, v2, e1, e2):
        dv1 = np.sqrt(v1[INDEX_FOR_X] ** 2 + v1[INDEX_FOR_Y] ** 2)
        dv2 = np.sqrt(v2[INDEX_FOR_X] ** 2 + v2[INDEX_FOR_Y] ** 2)
        dotv1v2 = v1[INDEX_FOR_X] * v2[INDEX_FOR_X] + v1[INDEX_FOR_Y] * v2[INDEX_FOR_Y]
        v1v2 = dv1 * dv2
        if v1v2 == 0:
            v1v2 = 1
        if abs(dotv1v2) == 0:
            dotv1v2 = 0
        cos_theta = round(dotv1v2 / v1v2, self.point_precision)
        sep = self.separation(e1, e2)
        return cos_theta, sep

    def get_linear_points(self, intersections):
        linear_ridges = []
        for intersect in intersections:
            p1 = intersect[0][0]
            p2 = intersect[0][1]
            p3 = intersect[1][1]
            if self.D(p2, p3) > self.lidar_fov and self.collinear(p1, p2, p3):
                linear_ridges.append(intersect)
        return linear_ridges

    def collinear(self, p1, p2, p3, width):
        s1 = self.slope(p1, p2)
        s2 = self.slope(p2, p3)
        if s1 == s2 or self.W(p2, p3) < width:
            return True
        return False

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

    def separation(self, e1, e2):
        p1 = e1[0]
        p2 = e1[1]
        p3 = e2[0]
        p4 = e2[1]
        c1 = p1[INDEX_FOR_Y] - self.slope(p1, p2) * p1[INDEX_FOR_X]
        c2 = p4[INDEX_FOR_Y] - self.slope(p3, p4) * p4[INDEX_FOR_X]
        return abs(c1 - c2)


    def is_free(self, p):
        xc = np.round(p[INDEX_FOR_X])
        yc = np.round(p[INDEX_FOR_Y])
        new_p = [0] * 2
        new_p[INDEX_FOR_X] = xc
        new_p[INDEX_FOR_Y] = yc
        new_p = tuple(new_p)
        return new_p in self.pixel_desc and self.pixel_desc[new_p] == FREE

    def neighbors(self, pose, scale):
        x = pose[INDEX_FOR_X]
        y = pose[INDEX_FOR_Y]
        neighbors = []
        for i in range(scale):
            ranges = [(x - i, y - i), (x - i, y + i), (x + i, y + i), (x + i, y - i)]
            for r in ranges:
                if r in self.pixel_desc and self.pixel_desc[r] != OCCUPIED:
                    neighbors.append(self.pixel_desc[r])
        if neighbors:
            return max(neighbors, key=neighbors.count) == FREE
        return False

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
        current_x = []
        current_y = []
        if ridge:
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

    def plot_data(self, frontiers, is_initial=False, vertext=None):
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xticks(fontsize=FONT_SIZE)
        plt.yticks(fontsize=FONT_SIZE)
        ax.set_xlabel("X", fontsize=FONT_SIZE)
        ax.set_ylabel("Y", fontsize=FONT_SIZE)
        ax.tick_params(labelsize=FONT_SIZE)

        unknown_points = []
        pixels = list(self.pixel_desc)
        for p in pixels:
            v = self.pixel_desc[p]
            if v != FREE and v != OCCUPIED:
                unknown_points.append(p)
        UX, UY = zip(*unknown_points)
        ax.scatter(UY, UX, marker='1', color='gray')
        obstacles = list(self.old_obstacles)
        leaves = list(self.old_leaf_slope)
        xr = [v[INDEX_FOR_X] for v in obstacles]
        yr = [v[INDEX_FOR_Y] for v in obstacles]
        ax.scatter(xr, yr, color='black', marker="1")
        x_pairs, y_pairs = self.process_edges()
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "g-o")
        if vertext:
            ax.scatter(vertext[INDEX_FOR_X], vertext[INDEX_FOR_Y], color='purple', marker=">")
        for leaf in leaves:
            ax.scatter(leaf[INDEX_FOR_X], leaf[INDEX_FOR_Y], color='red', marker='*', s=MARKER_SIZE)
        if is_initial:
            for leaf in leaves:
                if leaf in self.known_points and leaf in self.unknown_points:
                    ks = self.known_points[leaf]
                    us = self.unknown_points[leaf]
                    ux = [p[INDEX_FOR_X] for p in us]
                    uy = [p[INDEX_FOR_Y] for p in us]
                    ax.scatter(ux, uy, color='purple', marker="3")
                    if leaf not in frontiers:
                        ax.scatter(leaf[INDEX_FOR_X], leaf[INDEX_FOR_Y], color='red', marker='*')

            fx = [p[INDEX_FOR_X] for p in frontiers]
            fy = [p[INDEX_FOR_Y] for p in frontiers]
            ax.scatter(fx, fy, color='goldenrod', marker="D", s=MARKER_SIZE)
        plt.grid()
        plt.savefig("plots/plot_{}_{}.png".format(self.robot_id, time.time()))
        plt.close()
        # plt.show()

    def get_image_desc(self, polygon):
        self.lock.acquire()
        height = self.grid_values.shape[0]
        width = self.grid_values.shape[1]
        self.obstacles.clear()
        self.free_points.clear()
        for row in range(height):
            for col in range(width):
                index = [0] * 2
                index[INDEX_FOR_X] = col
                index[INDEX_FOR_Y] = row
                index = tuple(index)
                if self.in_range(index, polygon):
                    p = self.grid_values[row, col]
                    if p == FREE:
                        self.pixel_desc[index] = FREE
                        self.free_points[index] = FREE
                    elif p == OCCUPIED:
                        self.pixel_desc[index] = OCCUPIED
                        self.obstacles[index] = OCCUPIED
                    elif p == UNKNOWN:
                        self.pixel_desc[index] = UNKNOWN
        if self.obstacles:
            self.old_obstacles = copy.deepcopy(self.obstacles)
        self.lock.release()

    def in_range(self, point, polygon):
        x = point[INDEX_FOR_X]
        y = point[INDEX_FOR_Y]
        return polygon[0][INDEX_FOR_X] <= x <= polygon[2][INDEX_FOR_X] and polygon[0][INDEX_FOR_Y] <= y <= polygon[2][
            INDEX_FOR_Y]

    def create_polygon(self, pose):
        x = pose[INDEX_FOR_X]
        y = pose[INDEX_FOR_Y]
        rospy.logerr("Pose and range : {}, {}".format((x, y), self.comm_range))
        first = [0] * 2
        second = [0] * 2
        third = [0] * 2
        fourth = [0] * 2

        # if self.computing_for_frontier:
        #     first[INDEX_FOR_X] = 0
        #     first[INDEX_FOR_Y] = 0
        #
        #     second[INDEX_FOR_X] = 0
        #     second[INDEX_FOR_Y] = self.height
        #
        #     third[INDEX_FOR_X] = self.width
        #     third[INDEX_FOR_Y] = self.height
        #
        #     fourth[INDEX_FOR_X] = self.width
        #     fourth[INDEX_FOR_Y] = 0
        # else:
        first[INDEX_FOR_X] = x - self.comm_range
        first[INDEX_FOR_Y] = y - self.comm_range

        second[INDEX_FOR_X] = x - self.comm_range
        second[INDEX_FOR_Y] = y + self.comm_range

        third[INDEX_FOR_X] = x + self.comm_range
        third[INDEX_FOR_Y] = y + self.comm_range

        fourth[INDEX_FOR_X] = x + self.comm_range
        fourth[INDEX_FOR_Y] = y - self.comm_range

        ranges = [first, second, third, fourth]
        rospy.logerr("Polygon: {}".format(ranges))
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
