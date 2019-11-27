#!/usr/bin/python
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

FREE = 0
OCCUPIED = 100
UNKNOWN = 140
SMALL = 0.000000000001
INITIAL_SIZE = 6
MIN_WIDTH = 1.0
PRECISION = 1
EDGE_LENGTH = MIN_WIDTH  # / 4.0
SCAN_RADIUS = 5.0
THETA_SC = 360
PI = math.pi
ERROR_MARGIN = 0.03
RANGE_MARGIN = 1.0


class Graph:
    def __init__(self):
        self.call_count = 0
        self.obstacles = {}
        self.unknowns = {}
        self.free_points = {}
        self.resolution = None
        self.width = None
        self.height = None

    def load_image_data(self, file_path):
        im = Image.open(file_path, 'r')
        pix_map = im.load()
        width = im.size[0] - 1
        height = im.size[1] - 1
        for i in range(im.size[0]):
            for j in range(im.size[1]):
                py = height - j
                pixel = pix_map[i, j]
                point = (float(i), float(py))
                if pixel == 0:
                    self.obstacles[point] = []
                elif pixel == 140:
                    self.unknowns[point] = []
                else:
                    self.free_points[point] = []
        return width, height

    def read_raw_image(self, file_path):
        im = Image.open(file_path, 'r')
        self.resolution = 1
        basewidth = im.size[0] // 10
        wpercent = (basewidth / float(im.size[0]))
        hsize = int((float(im.size[1]) * float(wpercent)))
        im = im.resize((basewidth, hsize), Image.ANTIALIAS)

        pixelMap = im.load()
        img = Image.new(im.mode, im.size)
        pixelsNew = img.load()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                pixel = pixelMap[i, j]
                if i >= 20 and i < 40 and j >= 15:
                    pixelsNew[i, j] = 140
                else:
                    if pixel < 140:
                        pixelsNew[i, j] = 0
                    else:
                        pixelsNew[i, j] = 255
        self.obstacles.clear()
        self.free_points.clear()
        self.unknowns.clear()
        img.save('office_d_binary.png')

    def update_occupacygrid(self, occ_grid):
        resolution = occ_grid.info.resolution
        width = occ_grid.info.width
        height = occ_grid.info.height
        origin_pos = occ_grid.info.origin.position
        grid_values = occ_grid.data
        origin_x = origin_pos.x
        origin_y = origin_pos.y
        self.resolution = resolution
        self.width = width
        self.height = height
        for col in range(width):
            map_x = round(origin_x + col * resolution, PRECISION)
            for row in range(height):
                map_y = round(origin_y + row * resolution, PRECISION)
                data_val = grid_values[row * width + col]
                p = (map_x, map_y)
                neighbor_points = self.next_states(col, row)
                neighbors = []
                for n in neighbor_points:
                    x = n[0]
                    y = n[1]
                    if x < width and y < height:
                        xp = round(origin_x + x * resolution, PRECISION)
                        yp = round(origin_y + y * resolution, PRECISION)
                        xrp = (xp, yp)
                        neighbors.append(xrp)
                try:
                    if data_val == OCCUPIED:
                        self.obstacles[p] = neighbors

                    if data_val == UNKNOWN:
                        self.unknowns[p] = neighbors

                    if data_val == FREE:
                        self.free_points[p] = neighbors
                except Exception as e:
                    print("map update error: {}".format(e))
                    pass

    def next_states(self, col, row):
        right = (col + 1, row)
        down = (col, row - 1)
        up = (col, row + 1)
        left = (col - 1, row)
        return [right, down, up, left]

    def get_frontiers(self, count):
        frontiers = []
        vor, obstacles, adj_list, leaves, edges, ridges = self.compute_hallway_points()
        new_information, known_points, unknown_points = self.compute_new_information(ridges, leaves)
        while len(new_information) > 0:
            best_point = max(new_information, key=new_information.get)
            frontiers.append(best_point)
            if len(frontiers) == count:
                break
            del new_information[best_point]
        if len(frontiers) < count:
            frontiers = self.get_closest_unknown_region()
        return frontiers

    def compute_new_information(self, ridges, leaves):
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
                    ks = [p for p in area_points if p not in self.unknowns]
                    us = [p for p in area_points if p in self.unknowns]
                    full_area = len(area_points) * self.resolution ** 2
                    known_area = len(ks) * self.resolution ** 2
                    new_information[leaf] = full_area - known_area
                    known_points[leaf] = ks
                    unknown_points[leaf] = us
        return new_information, known_points, unknown_points

    def theta(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return math.atan2(dy, dx)

    def D(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def T(self, p, q):
        return self.D(p, q) * math.cos(self.theta(p, q))

    def W(self, p, q):
        return abs(self.D(p, q) * math.sin(self.theta(p, q)))

    def slope(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        if dx == 0:
            dx = SMALL
        return dy / dx

    def area(self, P, point):
        orientation = self.theta(P[0], P[1])
        points = []
        for d in np.arange(0, SCAN_RADIUS, self.resolution):
            distance_points = []
            for theta in range(-1 * THETA_SC // 2, THETA_SC + 1):
                angle = np.deg2rad(theta) + orientation
                x = point[0] + d * np.cos(angle)
                y = point[1] + d * np.sin(angle)
                p = self.round_point((x, y))
                distance_points.append(p)
            points += list(set(distance_points))
        return points

    def get_closest_unknown_region(self):
        known_cells = list(self.free_points)
        frontier_points = []
        for k in known_cells:
            neighbors = self.free_points[k]
            if any([p for p in neighbors if p in self.unknowns]):
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
        points = []
        count = 0
        for i in range(len(ridge_vertices)):
            ridge_vertex = ridge_vertices[i]
            ridge_point = ridge_points[i]
            if ridge_vertex[0] != -1 and ridge_vertex[1] != -1:
                p1 = tuple(vertices[ridge_vertex[0]])
                p2 = tuple(vertices[ridge_vertex[1]])
                if self.is_free(p1) and self.is_free(p2):
                    q1 = obstacles[ridge_point[0]]
                    q2 = obstacles[ridge_point[1]]
                    if self.D(q1, q2) > MIN_WIDTH:
                        r = ((p1, p2), (q1, q2))
                        hallway_vertices += [(p1, p2)]
                        hallway_edges.append(r)
                        points += list((p1, p2))
                        count += 1
        adj_list, leaf_nodes = self.get_adjacency_list(hallway_vertices)
        final_adj_list, edges, leaves = self.create_subgraph(adj_list, leaf_nodes)
        leaves = [p for p in leaves if self.is_free(p)]
        return vor, obstacles, final_adj_list, leaves, edges, hallway_edges

    def create_subgraph(self, adj_list, nodes):
        trees = {}
        tree_size = {}
        leave_dict = {}
        for s in nodes:
            S = [s]
            visited = []
            parents = {s: None}
            lf = []
            while len(S) > 0:
                u = S.pop()
                if u in adj_list:
                    neighbors = adj_list[u]
                    if len(neighbors) == 1:
                        lf.append(u)
                    for v in neighbors:
                        if v not in visited:
                            S.append(v)
                            parents[v] = u
                    visited.append(u)

            trees[s] = visited
            tree_size[s] = len(visited)
            leave_dict[s] = lf
        longest = max(tree_size, key=tree_size.get)
        longest_tree = trees[longest]
        leaves = []
        new_adj_list = {}
        edges = []
        for u in longest_tree:
            if u in adj_list:
                neigbours = adj_list[u]
                new_adj_list[u] = neigbours
                edges += [(u, v) for v in neigbours if (v, u) not in edges]
                if len(neigbours) == 1:
                    leaves.append(neigbours[0])

        return new_adj_list, edges, leaves

    def clean_ridge(self, ridge):
        p1 = (round(ridge[0][0][0], PRECISION), round(ridge[0][0][1], PRECISION))
        p2 = (round(ridge[0][1][0], PRECISION), round(ridge[0][1][1], PRECISION))
        q1 = (round(ridge[1][0][0], PRECISION), round(ridge[1][0][1], PRECISION))
        q2 = (round(ridge[1][1][0], PRECISION), round(ridge[1][1][1], PRECISION))
        return (p1, p2), (q1, q2)

    def round_point(self, p):
        return (round(p[0], PRECISION), round(p[1], PRECISION))

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
                        if -1 - ERROR_MARGIN <= cos_theta <= ERROR_MARGIN - 1 and self.D(p2,
                                                                                         p3) > SCAN_RADIUS and self.collinear(
                            p1, p2, p3):
                            intersections.append((i[0], j[0]))
        return intersections

    def compute_similarity(self, v1, v2):
        dv1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
        dv2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)
        dotv1v2 = v1[0] * v2[0] + v1[1] * v2[1]
        v1v2 = dv1 * dv2
        if v1v2 == 0:
            v1v2 = 1
        cos_theta = round(dotv1v2 / v1v2, 3)
        # print(cos_theta)
        separation = dv1 * math.sin(math.acos(cos_theta))
        return cos_theta, separation

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
                v = (p2[0] - p1[0], p2[1] - p1[1])
                width = self.D(q1, q2) #round(self.D(q1, q2), PRECISION)
                desc = (v, width)
                vertex_dict[r] = desc
                distance1 = self.D(p1, robot_pose)
                closest_ridge[distance1] = p1
                ri_dict[p1] = r
        close_point = closest_ridge[min(closest_ridge.keys())]
        corresponding_ridge = ri_dict[close_point]
        intersections = self.process_decision(vertex_dict, corresponding_ridge)
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

    def collinear(self, p1, p2, p3):
        s1 = self.slope(p1, p2)
        s2 = self.slope(p2, p3)
        if (self.D(p1, p3) - self.D(p2, p3) == self.D(p1, p2)) < self.D(p2, p3) and s1 == s2:
            return True
        return False

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
        xf = np.floor(p[0])
        yf = np.floor(p[1])
        xc = np.ceil(p[0])
        yc = np.ceil(p[1])
        return p in self.free_points or (xf, yf) in self.free_points or (xc, yc) in self.free_points

    def reject_outliers(self, data):
        raw_x = [v[0] for v in data]
        raw_y = [v[1] for v in data]
        rejected_points = [v for v in raw_x if v < 0]
        indexes = [i for i in range(len(raw_x)) if raw_x[i] in rejected_points]
        x_values = [raw_x[i] for i in range(len(raw_x)) if i not in indexes]
        y_values = [raw_y[i] for i in range(len(raw_y)) if i not in indexes]
        return x_values, y_values

    def plot_intersections(self, ax, ridge, intersections, obstacles, hallway_vertices, point, file_name):
        unknowns = list(self.unknowns)
        if not ax:
            fig, ax = plt.subplots(figsize=(16, 10))

        ux = [v[0] for v in unknowns]
        uy = [v[1] for v in unknowns]

        x = [v[0] for v in obstacles]
        y = [v[1] for v in obstacles]
        current_x = [ridge[0][0], ridge[1][0]]
        current_y = [ridge[0][1], ridge[1][1]]
        target_x = []
        target_y = []

        for section in intersections:
            Q = section[1]
            target_x += [Q[0][0], Q[1][0]]
            target_y += [Q[0][1], Q[1][1]]
        for i in range(len(target_x)):
            ax.plot(target_x[i], target_y[i], "g-d")

        for i in range(len(current_x)):
            ax.plot(current_x[i], current_y[i], "r-d")

        ax.scatter(point[0], point[1], marker='*', color='purple')
        ax.scatter(x, y, color='black', marker='s')
        ax.scatter(ux, uy, color='gray', marker='1')

        x_pairs, y_pairs = self.process_edges(hallway_vertices)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "m-.")

        plt.savefig("plots/{}_{}_{}.png".format(file_name, point[0], point[1]))
        # plt.show()

    def plot_data(self, ax, fp, leaves, edges, obstacles, known_points, unknown_points, is_initial=False, vertext=None):
        if not ax:
            fig, ax = plt.subplots(figsize=(16, 10))

        ux = [v[0] for v in unknown_points]
        uy = [v[1] for v in unknown_points]

        xr = [v[0] for v in obstacles]
        yr = [v[1] for v in obstacles]
        ax.scatter(xr, yr, color='black', marker="s")
        x_pairs, y_pairs = self.process_edges(edges)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "g-.")
        ax.scatter(fp[0], fp[1], color='blue', marker='s')

        if vertext:
            ax.scatter(vertext[0], vertext[1], color='purple', marker=">")
        for leaf in leaves:
            ax.scatter(leaf[0], leaf[1], color='red', marker='*')
        ax.scatter(ux, uy, color='gray', marker='1')
        if is_initial:
            for leaf in leaves:
                if leaf in known_points and leaf in unknown_points:
                    ks = known_points[leaf]
                    us = unknown_points[leaf]
                    kx = [p[0] for p in ks]
                    ky = [p[1] for p in ks]
                    ux = [p[0] for p in us]
                    uy = [p[1] for p in us]
                    ax.scatter(ux, uy, color='purple', marker="3")
                    ax.scatter(kx, ky, color='gray', marker="3")
                    ax.scatter(leaf[0], leaf[1], color='red', marker='*')

        plt.show()

if __name__ == "__main__":
    graph = Graph()
