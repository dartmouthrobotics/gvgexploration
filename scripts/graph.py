#!/usr/bin/python
import copy
import operator
import matplotlib
# matplotlib.use('Agg')
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

FREE = 0
OCCUPIED = 100
UNKNOWN = 140
UNKNOWN_PIX = -1
SMALL = 0.000000000001
INITIAL_SIZE = 6
MIN_WIDTH = 0.5 / 0.05
PRECISION = 1
EDGE_LENGTH = MIN_WIDTH  # / 4.0
SCAN_RADIUS = 5.0 / 0.1
THETA_SC = 360
PI = math.pi
ERROR_MARGIN = 0.03
RANGE_MARGIN = 1.0
INDEX_FOR_X = 1
INDEX_FOR_Y = 0
SLOPE_MARGIN = 0.05


class Graph:
    def __init__(self):
        self.call_count = 0
        self.obstacles = {}
        self.unknowns = {}
        self.free_points = {}
        self.resolution = None
        self.width = None
        self.height = None
        self.point = None
        self.origin_x = None
        self.origin_y = None

    def update_occupacygrid(self, occ_grid):
        resolution = occ_grid.info.resolution
        width = occ_grid.info.width
        height = occ_grid.info.height
        origin_pos = occ_grid.info.origin.position
        grid_values = list(occ_grid.data)
        self.origin_x = origin_pos.x
        self.origin_y = origin_pos.y
        self.resolution = resolution
        grid_values = np.array(grid_values).reshape((height, width)).astype(np.float32)
        pixel_desc = self.get_image_desc(grid_values)
        self.get_free_points(pixel_desc)
        self.get_obstacles(pixel_desc)
        self.get_unknowns(pixel_desc)

    def get_frontiers(self, pose, count):
        x = round((pose[INDEX_FOR_X] - self.origin_x) / self.resolution, 2)
        y = round((pose[INDEX_FOR_Y] - self.origin_y) / self.resolution, 2)
        pose=[0]*2
        pose[INDEX_FOR_X]= x
        pose[INDEX_FOR_Y]=y
        start = time.time()
        frontiers = []
        vor, obstacles, adj_list, leaves, edges, ridges = self.compute_hallway_points()
        new_information, known_points, unknown_points = self.compute_new_information(ridges, leaves)
        while len(new_information) > 0:
            best_point = max(new_information, key=new_information.get)
            frontiers.append(best_point)
            if len(frontiers) == count:
                break
            del new_information[best_point]
        # if len(frontiers) < count:
        #     frontiers = self.get_closest_unknown_region()
        actual_points=[]
        for p in frontiers:
            new_p = [0] * 2
            new_p[INDEX_FOR_X] = round(self.origin_x + p[INDEX_FOR_X] * self.resolution, 2)
            new_p[INDEX_FOR_Y] = round(self.origin_x + p[INDEX_FOR_Y] * self.resolution, 2)
            actual_points.append(tuple(new_p))
        rospy.logerr("Total time: {}".format(time.time() - start))
        self.plot_data(None, pose, leaves, edges, obstacles, known_points, unknown_points, frontiers, is_initial=True)

        return actual_points

    def compute_intersections(self, robot_pose):
        vor, obstacles, adj_list, leaves, edges, ridges = self.compute_hallway_points()
        vertex_dict = {}
        ri_dict = {}
        closest_ridge = {}
        for r in ridges:
            p1 = r[0][0]
            p2 = r[0][1]
            q1 = r[1][0]
            q2 = r[1][1]
            u_check = self.W(p1, robot_pose) < self.D(q1, q2) or self.W(p2, robot_pose) < self.D(q1, q2)
            if u_check:
                v =[]*2
                v[INDEX_FOR_X]=p2[INDEX_FOR_X] - p1[INDEX_FOR_X]
                v[INDEX_FOR_Y]=p2[INDEX_FOR_Y] - p1[INDEX_FOR_Y]
                width = self.D(q1, q2)
                desc = (tuple(v), width)
                vertex_dict[r] = desc
                distance1 = self.D(p1, robot_pose)
                closest_ridge[distance1] = p1
                ri_dict[p1] = r
        close_point = closest_ridge[min(closest_ridge.keys())]
        corresponding_ridge = ri_dict[close_point]
        intersections = self.process_decision(vertex_dict, corresponding_ridge)
        return (corresponding_ridge[0][0], corresponding_ridge[0][1]), intersections

    def compute_new_information(self, ridges, leaves):
        resolution = 1
        frontier_size = {}
        new_information = {}
        known_points = {}
        unknown_points = {}
        for leaf in leaves:
            for r in ridges:
                P = r[0]
                Q = r[1]
                if leaf in P:
                    frontier_size[leaf] = self.W(leaf, Q[0])
                    area_points = self.area(P, leaf)
                    ks = [p for p in area_points if self.is_free(p)]
                    us = [p for p in area_points if not self.is_free(p) and not self.is_obstacle(p)]
                    full_area = len(area_points) * resolution ** 2
                    known_area = len(ks) * resolution ** 2
                    new_information[leaf] = full_area - known_area
                    known_points[leaf] = ks
                    unknown_points[leaf] = us
        return new_information, known_points, unknown_points

    def next_states(self, row, col, pixel_dict, scale=10):
        neighbors = []
        for i in range(scale):
            values = []
            right = [0] * 2
            down = [0] * 2
            up = [0] * 2
            left = [0] * 2
            right[INDEX_FOR_X] = row + i
            right[INDEX_FOR_Y] = col

            down[INDEX_FOR_X] = row
            down[INDEX_FOR_Y] = col - 1

            up[INDEX_FOR_X] = row
            up[INDEX_FOR_Y] = col + 1

            left[INDEX_FOR_X] = row - i
            left[INDEX_FOR_Y] = col

            if tuple(right) in pixel_dict:
                values.append(pixel_dict[tuple(right)])
            if tuple(down) in pixel_dict:
                values.append(pixel_dict[tuple(down)])
            if tuple(up) in pixel_dict:
                values.append(pixel_dict[tuple(up)])
            if tuple(left) in pixel_dict:
                values.append(pixel_dict[tuple(left)])
            neighbors += values
        return neighbors

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

    def area(self, P, point):
        resolution = 1
        orientation = self.theta(P[0], P[1])
        points = []
        for d in np.arange(0, SCAN_RADIUS, resolution):
            distance_points = []
            for theta in range(-1 * THETA_SC // 2, THETA_SC + 1):
                angle = np.deg2rad(theta) + orientation
                x = point[INDEX_FOR_X] + d * np.cos(angle)
                y = point[INDEX_FOR_Y] + d * np.sin(angle)
                new_p = [0] * 2
                new_p[INDEX_FOR_X] = x
                new_p[INDEX_FOR_Y] = y
                distance_points.append(tuple(new_p))
            points += list(set(distance_points))
        return points

    def get_closest_unknown_region(self,count):
        known_cells = list(self.free_points)
        frontier_points = []
        for k in known_cells:
            neighbors = self.free_points[k]
            if neighbors and max(neighbors, key=neighbors.count) == UNKNOWN_PIX:
                frontier_points.append(k)
        return frontier_points

    def compute_hallway_points(self):
        obstacles = list(self.obstacles)
        vor = Voronoi(obstacles)
        vertices = vor.vertices
        ridge_vertices = vor.ridge_vertices
        ridge_points = vor.ridge_points
        hallway_vertices = []
        hallway_edges = []
        count = 0
        for i in range(len(ridge_vertices)):
            ridge_vertex = ridge_vertices[i]
            ridge_point = ridge_points[i]
            if ridge_vertex[0] != -1 and ridge_vertex[1] != -1:
                p1 = [0] * 2
                p2 = [0] * 2
                p1[INDEX_FOR_X] = round(vertices[ridge_vertex[0]][INDEX_FOR_X], 2)
                p1[INDEX_FOR_Y] = round(vertices[ridge_vertex[0]][INDEX_FOR_Y], 2)
                p2[INDEX_FOR_X] = round(vertices[ridge_vertex[1]][INDEX_FOR_X], 2)
                p2[INDEX_FOR_Y] = round(vertices[ridge_vertex[1]][INDEX_FOR_Y], 2)
                if self.is_free(p1) and self.is_free(p2):
                    q1 = obstacles[ridge_point[0]]
                    q2 = obstacles[ridge_point[1]]
                    if self.D(q1, q2) > MIN_WIDTH:
                        r = ((tuple(p1), tuple(p2)), (q1, q2))
                        hallway_vertices += [(tuple(p1), tuple(p2))]
                        hallway_edges.append(r)
                        count += 1
        adj_list, leaf_nodes = self.get_adjacency_list(hallway_vertices)
        adj_list, leaf_nodes, hallway_vertices = self.connect_subtrees(adj_list, leaf_nodes)
        adj_list, leaf_nodes, hallway_vertices = self.merge_similar_edges(adj_list, leaf_nodes)
        return vor, obstacles, adj_list, leaf_nodes, hallway_vertices, hallway_edges

    def merge_similar_edges(self, adj_list, nodes):
        longest, adj_dict, tree_size, leave_dict, parent_dict = self.get_deeepest_tree_source(adj_list, nodes)
        tree_leaves = leave_dict[longest]
        parents = parent_dict[longest]
        R = []
        visited = []
        for leaf in tree_leaves:
            u = parents[leaf]
            while u is not None:
                us = self.slope(u, leaf)
                if parents[u] is None:
                    break
                else:
                    if u not in visited:
                        ps = self.slope(parents[u], u)
                        if abs(us + ps) > SLOPE_MARGIN:
                            if parents[u] is not None:
                                if (u, leaf) not in R:
                                    R.append((u, leaf))
                                    leaf = u
                                pars = copy.deepcopy(parents)
                                for k, p in pars.items():
                                    if p == leaf and k not in visited:
                                        parents[k] = u
                        visited.append(u)
                        u = parents[u]
                    else:
                        R.append((u, leaf))
                        break

        adjlist, leaves = self.get_adjacency_list(R)
        return adjlist, leaves, R

    def connect_subtrees(self, adj_list, nodes):
        merged_adj = copy.deepcopy(adj_list)
        longest, adj_dict, tree_size, leave_dict, parent_dict = self.get_deeepest_tree_source(adj_list, nodes)
        if len(adj_list) == len(adj_dict[longest]):
            edges = []
            leaves = []
            full_tree = adj_dict[longest]
            for k, v in full_tree.items():
                if len(v) == 1:
                    leaves.append(k)
                edges += [(k, q) for q in v]
            return merged_adj, leaves, edges
        rospy.logerr("ADJ: {} merged: {}".format(len(adj_list), len(adj_dict[longest])))
        allnodes = list(merged_adj)
        for leaf, adj in adj_dict.items():
            adj_leaves = leave_dict[leaf]
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
                merged_adj[closest].append(close_node)
                merged_adj[close_node].append(closest)
        leaves = []
        for k, v in merged_adj.items():
            if len(v) == 1:
                leaves.append(k)

        return self.connect_subtrees(merged_adj, leaves)

    def get_deeepest_tree_source(self, adj_list, nodes):
        trees = {}
        leave_dict = {}
        adj_dict = {}
        tree_size = {}
        parent_dict = {}
        for s in nodes:
            S = [s]
            visited = []
            parents = {s: None}
            lf = []
            node_adjlist = {}
            while len(S) > 0:
                u = S.pop()
                if u in adj_list:
                    neighbors = adj_list[u]
                    if u not in node_adjlist:
                        node_adjlist[u] = neighbors

                    if len(neighbors) == 1:
                        lf.append(u)
                    for v in neighbors:
                        if v not in visited:
                            S.append(v)
                            parents[v] = u
                    visited.append(u)
            adj_dict[s] = copy.deepcopy(node_adjlist)
            trees[s] = copy.deepcopy(visited)
            tree_size[s] = len(visited)
            leave_dict[s] = copy.deepcopy(lf)
            parent_dict[s] = parents
        longest = max(tree_size, key=tree_size.get)
        return longest, adj_dict, tree_size, leave_dict, parent_dict

    def get_close_ridge(self, fp):
        vor, obstacles, adj_list, leaves, edges, ridges = self.compute_hallway_points()

        leaves = [k for k, p in adj_list.items() if len(p) == 1]
        close_ridges = []

        for r in ridges:
            p1 = r[0][0]
            p2 = r[0][1]
            q1 = r[1][0]
            q2 = r[1][1]
            u_check = self.W(p1, fp) < self.D(q1, q2) or self.W(p2, fp) < self.D(q1, q2)
            if u_check:
                close_ridges.append(r)
        vertex = None
        vertex_dict = {}
        if close_ridges:
            closest_ridge = {}
            for P in close_ridges:
                distance1 = self.D(P[0][0], fp)
                closest_ridge[distance1] = P[0][0]
            vertex = closest_ridge[min(closest_ridge.keys())]
        if vertex in adj_list:
            neighbors = adj_list[vertex]
            for n in neighbors:
                if (vertex, n) in [r[0] for r in ridges]:
                    desc, new_r, new_edges, obs = self.compute_ridge_desc(vertex, n, ridges)
                    vertex_dict[(vertex, n)] = desc
        new_information, known_points, unknown_points = self.compute_new_information(ridges, leaves)

        self.plot_data(None, fp, leaves, edges, obstacles, known_points, unknown_points, is_initial=False,
                       vertext=vertex)
        return vertex_dict

    def compute_ridge_desc(self, p, q, ridges):
        connected_ridges = []
        pq_slope = self.slope(p, q)
        pq_ridge = [r for r in ridges if (p, q) == r[0]][0]
        vertices = []
        for r in ridges:
            if p == r[0][0] or q == r[0][0]:
                if self.slope(r[0][0], r[0][1]) == pq_slope:
                    vertices += [r[0][0], r[0][1]]
                    connected_ridges.append(r)
        furthest = {}
        for v in vertices:
            for v1 in vertices:
                furthest[(v, v1)] = self.D(v, v1)
        furthest_point = max(furthest, key=furthest.get)
        center = [0] * 2
        center[INDEX_FOR_X] = round((furthest_point[0][INDEX_FOR_X] + furthest_point[1][INDEX_FOR_X]) / 2.0, 2)
        center[INDEX_FOR_Y] = round((furthest_point[0][INDEX_FOR_Y] + furthest_point[1][INDEX_FOR_Y]) / 2.0, 2)
        p1 = furthest_point[0]
        p2 = furthest_point[1]
        q1 = pq_ridge[1][0]
        q2 = pq_ridge[1][1]
        q1 = (q1[INDEX_FOR_X] + self.T(q1, center), q1[INDEX_FOR_Y] + self.W(q1, center))
        q2 = (q2[INDEX_FOR_X] + self.T(q2, center), q2[INDEX_FOR_Y] + self.W(q2, center))
        new_r = ((p1, p2), (q1, q2))
        thet = round(self.theta(p1, p2), PRECISION)
        distance = round(self.D(p1, p2), PRECISION)
        slop = pq_slope
        width = round(self.W(q1, q2), PRECISION)
        desc = (thet, distance, slop, width)
        top_o = [0] * 2
        bot_o = [0] * 2
        top_o[INDEX_FOR_X] = q1[INDEX_FOR_X]
        top_o[INDEX_FOR_Y] = q1[INDEX_FOR_Y]
        bot_o[INDEX_FOR_X] = q2[INDEX_FOR_X]
        bot_o[INDEX_FOR_Y] = q2[INDEX_FOR_Y]
        return desc, new_r, (p1, p2), tuple(top_o), tuple(bot_o)

    def round_point(self, p):
        return (round(p[INDEX_FOR_X], PRECISION), round(p[INDEX_FOR_Y], PRECISION))

    def process_decision(self, vertex_descriptions, ridge):
        intersections = []
        for i, desci in vertex_descriptions.items():
            if ridge == i:
                v1 = desci[0]
                w1 = desci[1]
                for j, descj in vertex_descriptions.items():
                    if i != j:
                        descj = vertex_descriptions[j]
                        v2 = descj[0]
                        w2 = descj[1]
                        cos_theta, separation = self.compute_similarity(v1, v2)
                        p1 = i[0][0]
                        p2 = i[0][1]
                        p3 = j[0][1]
                        width_diff = abs(w1 - w2)
                        if self.D(p2, p3) > SCAN_RADIUS:
                            if -1 <= cos_theta <= -0.9:
                                if separation <= 0.1:
                                    if width_diff < 1:
                                        if self.collinear(p1, p2, p3, w1):
                                            if self.there_is_unknown_region(p2, p3):
                                                intersections.append((i[0], j[0]))
        return intersections

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

    def check_data_sharing_status(self, point):
        x = round((point[INDEX_FOR_X] - self.origin_x) / self.resolution, 2)
        y = round((point[INDEX_FOR_Y] - self.origin_y) / self.resolution, 2)
        pose=[0]*2
        pose[INDEX_FOR_X]=x
        pose[INDEX_FOR_Y]=y
        robot_pose = tuple(pose)
        vor, obstacles, adj_list, leaves, edges, ridges = self.compute_hallway_points()
        vertex_dict = {}
        ri_dict = {}
        closest_ridge = {}
        for r in ridges:
            p1 = r[0][0]
            p2 = r[0][1]
            q1 = r[1][0]
            q2 = r[1][1]
            v =[0]
            v[INDEX_FOR_X]=p2[INDEX_FOR_X] - p1[INDEX_FOR_X]
            v[INDEX_FOR_Y]=p2[INDEX_FOR_Y] - p1[INDEX_FOR_Y]  # TODO check if this is correct.
            width = self.D(q1, q2)  # round(self.D(q1, q2), PRECISION)
            desc = (tuple(v), width)
            vertex_dict[r] = desc
            distance1 = self.D(p1, robot_pose)
            closest_ridge[distance1] = p1
            ri_dict[p1] = r
        close_point = closest_ridge[min(closest_ridge.keys())]
        corresponding_ridge = ri_dict[close_point]
        intersections = self.process_decision(vertex_dict, corresponding_ridge)
        self.plot_intersections(None, (corresponding_ridge[0][0], corresponding_ridge[0][1]), intersections, obstacles,
                                edges, robot_pose, 'general_file', pose)
        return (corresponding_ridge[0][0], corresponding_ridge[0][1]), intersections

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
                region_point = (x, y)
                if region_point in self.unknowns:
                    points.append(region_point)
        return len(points) >= point_count / 2.0

    def get_adjacency_list(self, hallway_vertices):
        adj_list = {}
        for e in hallway_vertices:
            first = e[0]
            second = e[1]
            if first in adj_list:
                adj_list[first].append(second)
            else:
                adj_list[first] = [second]
            if second in adj_list:
                adj_list[second].append(first)
            else:
                adj_list[second] = [first]
        nodes = [k for k, v in adj_list.items() if len(v) == 1]
        return adj_list, nodes

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

    def process_edges(self, edges):
        x_pairs = []
        y_pairs = []
        for edge in edges:
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
        return tuple(new_p) in self.free_points

    def is_obstacle(self, p):
        xc = np.round(p[INDEX_FOR_X])
        yc = np.round(p[INDEX_FOR_Y])
        new_p = [0] * 2
        new_p[INDEX_FOR_X] = xc
        new_p[INDEX_FOR_Y] = yc
        return tuple(new_p) in self.obstacles

    def get_obstacles(self, pixel_dict):
        pixels = list(pixel_dict)
        self.obstacles = {p: self.next_states(p[INDEX_FOR_X], p[INDEX_FOR_Y], pixel_dict) for p in pixels if
                          pixel_dict[p] == OCCUPIED}

    def get_free_points(self, pixel_dict):
        pixels = list(pixel_dict)
        self.free_points = {p: self.next_states(p[INDEX_FOR_X], p[INDEX_FOR_Y], pixel_dict) for p in pixels if
                            pixel_dict[p] == FREE}

    def get_unknowns(self, pixel_dict):
        pixels = list(pixel_dict)
        self.unknowns = {p: self.next_states(p[INDEX_FOR_X], p[INDEX_FOR_Y], pixel_dict) for p in pixels if
                         pixel_dict[p] == UNKNOWN_PIX}

    def reject_outliers(self, data):
        raw_x = [v[INDEX_FOR_X] for v in data]
        raw_y = [v[INDEX_FOR_Y] for v in data]
        rejected_points = [v for v in raw_x if v < 0]
        indexes = [i for i in range(len(raw_x)) if raw_x[i] in rejected_points]
        x_values = [raw_x[i] for i in range(len(raw_x)) if i not in indexes]
        y_values = [raw_y[i] for i in range(len(raw_y)) if i not in indexes]
        return x_values, y_values

    def plot_intersections(self, ax, ridge, intersections, obstacles, hallway_vertices, point, file_name, pose):
        unknowns = list(self.unknowns)
        if not ax:
            fig, ax = plt.subplots(figsize=(16, 10))

        ux = [v[INDEX_FOR_X] for v in unknowns]
        uy = [v[INDEX_FOR_Y] for v in unknowns]

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
        ax.scatter(ux, uy, color='gray', marker='1')

        ax.scatter(pose[INDEX_FOR_X], pose[INDEX_FOR_Y], color='magenta', marker='s')

        x_pairs, y_pairs = self.process_edges(hallway_vertices)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "m-.")

        plt.savefig("plots/intersections_{}.png".format(time.time()))
        # plt.show()

    def plot_data(self, ax, fp, leaves, edges, obstacles, known_points, unknown_points, frontiers, is_initial=False,
                  vertext=None):
        if not ax:
            fig, ax = plt.subplots(figsize=(16, 10))

        ux = [v[INDEX_FOR_X] for v in unknown_points]
        uy = [v[INDEX_FOR_Y] for v in unknown_points]

        xr = [v[INDEX_FOR_X] for v in obstacles]
        yr = [v[INDEX_FOR_Y] for v in obstacles]
        ax.scatter(xr, yr, color='black', marker="s")
        x_pairs, y_pairs = self.process_edges(edges)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "g-.")
        ax.scatter(fp[INDEX_FOR_X], fp[INDEX_FOR_Y], color='blue', marker='s')  # Pose in x, y -- col, row.

        if vertext:
            ax.scatter(vertext[INDEX_FOR_X], vertext[INDEX_FOR_Y], color='purple', marker=">")
        for leaf in leaves:
            ax.scatter(leaf[INDEX_FOR_X], leaf[INDEX_FOR_Y], color='red', marker='*')
        ax.scatter(ux, uy, color='gray', marker='1')
        if is_initial:
            for leaf in leaves:
                if leaf in known_points and leaf in unknown_points:
                    ks = known_points[leaf]
                    us = unknown_points[leaf]
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
        plt.savefig("plots/plot_{}.png".format(time.time()))
        # plt.show()

    def get_image_desc(self, grid_values):
        self.height = grid_values.shape[0]
        self.width = grid_values.shape[1]
        pixel_desc = {}
        for row in range(self.height):
            for col in range(self.width):
                p = grid_values[row, col]
                if p == -1:
                    pixel_desc[(row, col)] = UNKNOWN_PIX
                    grid_values[row, col] = 128  # TODO: have a "MACRO" for these values. # TODO this is just for debug.
                elif -1 < p <= 0:
                    pixel_desc[(row, col)] = FREE
                    grid_values[row, col] = 255
                elif p > 0:
                    pixel_desc[(row, col)] = OCCUPIED
                    grid_values[row, col] = 0
        cv2.imwrite("fullsize.png", grid_values)
        return pixel_desc


if __name__ == "__main__":
    graph = Graph()
    filename = "map_message1.pickle"
    # graph.load_map_data(filename)
