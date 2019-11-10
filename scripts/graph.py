#!/usr/bin/python
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rospy
import math
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

"""
    Node class for A* algorithm.
"""
SCALE = 15  # TODO as param
FREE = 0
OCCUPIED = 100
UNKNOWN = -1

ROBOT_SPACE = 1.0  # TODO pick from param (robot radius + inflation radius)

OUT_OF_BOUNDS = -2

SMALL = 0.000000000001
INITIAL_SIZE = 6
MIN_WIDTH = 1.0
PRECISION = 1
EDGE_LENGTH = MIN_WIDTH  # / 4.0
SCAN_RADIUS = 0.8
THETA_SC = 120
PI = math.pi
error_margin = 0.01
range_margin = 0.1


class Graph:

    def __init__(self):
        self.call_count = 0
        self.obstacles = {}
        self.unknowns = {}
        self.free_points = {}
        self.resolution = None
        self.width = None
        self.height = None
        # self.gvg = None

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
                    if data_val == OCCUPIED and p not in self.obstacles:
                        self.obstacles[p] = neighbors

                    if data_val == UNKNOWN and p not in self.unknowns:
                        self.unknowns[p] = neighbors

                    if data_val == FREE and p not in self.free_points:
                        self.free_points[p] = neighbors
                except Exception as e:
                    rospy.loginfo("map update error: {}".format(e))
                    pass

    def is_enough_data(self):
        if len(self.obstacles) < INITIAL_SIZE:
            return False
        keys = list(self.obstacles)
        ys = set([p[1] for p in keys])
        xs = set([p[0] for p in keys])
        return len(xs) > 1 and len(ys) > 1

    def next_states(self, col, row):
        right = (col + 1, row)
        down = (col, row - 1)
        up = (col, row + 1)
        left = (col - 1, row)
        return [right, down, up, left]

    def get_neighboring_states(self, pose):
        states = []
        x = round(pose[0], PRECISION)
        y = round(pose[1], PRECISION)
        p = (x, y)
        if p in self.free_points:
            states = self.free_points[p]
        elif p in self.obstacles:
            states = self.obstacles[p]
        elif p in self.unknowns:
            states = self.unknowns[p]
        return states

    # '''
    # Get the points which are closest to the unknown area
    # '''
    def get_frontiers(self, rid, pose, count):
        frontiers = []
        vor, obstacles, adj_list, edges, ridges = self.compute_hallway_points()
        leaves = [k for k, p in adj_list.items() if len(p) == 1]
        new_information, known_points, unknown_points = self.compute_new_information(ridges, leaves)
        while len(new_information) > 0:
            best_point = max(new_information, key=new_information.get)
            frontiers.append(best_point)
            if len(frontiers) == count:
                break
            del new_information[best_point]
        if len(frontiers) < count:
            frontiers = self.get_closest_unknown_region()
        self.plot_data(rid, pose, leaves, edges, vor, obstacles, known_points, unknown_points)
        return frontiers

    def compute_new_information(self, ridges, leaves):
        frontier_size = {}
        new_information = {}
        known_points = {}
        unknown_points = {}
        for leaf in leaves:
            if leaf in self.free_points:
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
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        if dx == 0:
            dx = SMALL
        return dy / dx

    def point_is_free(self, pose):
        return pose in self.free_points

    def area(self, P, point):
        dx = P[0][0] - P[1][0]
        dy = P[0][1] - P[1][1]
        if dx == 0:
            dx = SMALL
        orientation = math.atan2(dy, dx)

        points = []
        for d in np.arange(0, SCAN_RADIUS, self.resolution):
            distance_points = []
            for theta in range(THETA_SC + 1):
                x = point[0] + d * np.cos(np.deg2rad(theta - orientation))
                y = point[1] + d * np.sin(np.deg2rad(theta - orientation))
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
        for i in range(len(ridge_vertices)):
            ridge_vertex = ridge_vertices[i]
            ridge_point = ridge_points[i]
            if ridge_vertex[0] != -1 and ridge_vertex[1] != -1:
                p1 = vertices[ridge_vertex[0]]
                p2 = vertices[ridge_vertex[1]]
                if self.round_point(p1) in self.free_points or self.round_point(p2) in self.free_points:
                    q1 = obstacles[ridge_point[0]]
                    q2 = obstacles[ridge_point[1]]
                    if self.D(q1, q2) >= MIN_WIDTH:  # and self.D(p1, p2) >= EDGE_LENGTH:
                        r = self.clean_ridge([(p1, p2), (q1, q2)])
                        hallway_vertices += [r[0]]
                        hallway_edges.append(r)
                        points += list(r[0])
        adj_list,ridge_dict = self.get_adjacency_list(hallway_vertices)
        final_adj_list, edges = self.create_subgraph(adj_list, list(set(points)))
        return vor, obstacles, final_adj_list, edges, hallway_edges

    def create_subgraph(self, adj_list, nodes):
        trees = {}
        tree_size = {}
        for s in nodes:
            S = [s]
            visited = []
            parents = {s: None}
            while len(S) > 0:
                u = S.pop()
                if u in adj_list:
                    neighbors = adj_list[u]
                    for v in neighbors:
                        if v not in visited:
                            S.append(v)
                            parents[v] = u
                    visited.append(u)

            trees[s] = visited
            tree_size[s] = len(visited)

        longest = max(tree_size, key=tree_size.get)
        longest_tree = trees[longest]
        new_adj_list = {}
        edges = []
        for u in longest_tree:
            if u in adj_list:
                neigbours = adj_list[u]
                new_adj_list[u] = neigbours
                edges += [(u, v) for v in neigbours if (v, u) not in edges]
        return new_adj_list, edges

    def clean_ridge(self, ridge):
        p1 = (round(ridge[0][0][0], PRECISION), round(ridge[0][0][1], PRECISION))
        p2 = (round(ridge[0][1][0], PRECISION), round(ridge[0][1][1], PRECISION))
        q1 = (round(ridge[1][0][0], PRECISION), round(ridge[1][0][1], PRECISION))
        q2 = (round(ridge[1][1][0], PRECISION), round(ridge[1][1][1], PRECISION))
        return (p1, p2), (q1, q2)

    def round_point(self, p):
        return (round(p[0], PRECISION), round(p[1], PRECISION))

    def get_close_ridge(self, fp, rid):
        vor, obstacles, adj_list, edges, ridges = self.compute_hallway_points()
        leaves = [k for k, p in adj_list.items() if len(p) == 1]
        close_ridges = []
        for r in ridges:
            p1 = r[0][0]
            p2 = r[0][1]
            q1 = r[1][0]
            q2 = r[1][1]
            u_check = self.T(p1, fp) < self.T(p1, q1) or self.T(p1, fp) < self.T(p1, q2)
            v_check = self.T(p2, fp) < self.T(p2, q1) or self.T(p2, fp) < self.T(p2, q2)
            if u_check or v_check:
                close_ridges.append(r)
        vertex = None
        vertex_dict = {}
        if close_ridges:
            closest_ridge = {}
            for P in close_ridges:
                distance1 = self.D(P[0][0], fp)
                distance2 = self.D(P[0][1], fp)
                if distance1 <= distance2:
                    closest_ridge[distance1] = P[0][0]
                else:
                    closest_ridge[distance1] = P[0][1]
            vertex = closest_ridge[min(closest_ridge.keys())]

        if vertex and self.D(fp, vertex) <= range_margin:
            if vertex in adj_list:
                neighbors = adj_list[vertex]
                for n in neighbors:
                    desc = self.compute_robot_desc(fp, vertex,ridges)
                    vertex_dict[(vertex, n)] = desc
        new_information, known_points, unknown_points = self.compute_new_information(ridges, leaves)
        self.plot_data(rid, fp, leaves, edges, vor, obstacles, known_points, unknown_points)
        return vertex_dict

    def expand_ridge(self,P_raw, ridges):
        connected_ridges=[]
        for r in ridges:
            if P_raw[0] in r[0] or P_raw[1] in r[0]:
                connected_ridges.append(r)
        if connected_ridges:
            Px=[]
            Py=[]
            for c in connected_ridges:
                Px+=[c[0][0][0], c[0][1][0]]
                Py += [c[0][0][0], c[0][1][0]]

        return P_raw

    def plot_data(self, rid, fp, leaves, edges, vor, obstacles, known_points, unknown_points):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        voronoi_plot_2d(vor, ax=ax1)
        xr = [v[0] for v in obstacles]
        yr = [v[1] for v in obstacles]
        ax2.scatter(xr, yr, color='black', marker="s")
        # process edges
        x_pairs, y_pairs = self.process_edges(edges)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax2.plot(x, y, "g-o")
        ax2.scatter(fp[0], fp[1], color='blue', marker='s')

        for leaf in leaves:
            if leaf in known_points and leaf in unknown_points:
                ks = known_points[leaf]
                us = unknown_points[leaf]
                kx = [p[0] for p in ks]
                ky = [p[1] for p in ks]
                ux = [p[0] for p in us]
                uy = [p[1] for p in us]
                ax2.scatter(ux, uy, color='purple', marker="3")
                ax2.scatter(kx, ky, color='gray', marker="3")
                ax2.scatter(leaf[0], leaf[1], color='red', marker='*')

        plt.savefig('plots/plot_{}_{}.png'.format(rid, rospy.Time.now().secs))

    def process_decision(self, vertex_descriptions):
        intersections = []
        v_keys = list(vertex_descriptions)
        slope_dict = {}
        angle_dict = {}
        for k in v_keys:
            desc = vertex_descriptions[k]
            a = desc[0]
            d = desc[1]
            s = desc[2]
            w = desc[3]
            if a in angle_dict:
                angle_dict[a].append(k)
            else:
                angle_dict[a] = [k]
            if s in slope_dict:
                slope_dict[s].append(k)
            else:
                slope_dict[s] = [k]
        slope_common = []
        angle_common = []
        for p in slope_dict.keys():
            for q in slope_dict.keys():
                if p + q <= error_margin:
                    slope_common += slope_dict[p] + slope_dict[q]
        for p in angle_dict.keys():
            for q in angle_dict.keys():
                if PI - (p + q) <= error_margin:
                    angle_common += angle_dict[p] + angle_dict[q]
        if slope_common and angle_common:
            slope_set = set(slope_common)
            angle_set = set(angle_common)
            intersections = list(slope_set.intersection(angle_set))
        return intersections

    def compute_robot_desc(self, p, q,ridges):
        connected_vertices = self.expand_ridge(p, ridges)   # do a more robust computation
        theta = round(self.theta(p, q), PRECISION)
        distance = round(self.D(p, q), PRECISION)
        slope = round(self.slope(p, q), PRECISION)
        width = round(self.W(p, q), PRECISION)
        desc = (theta, distance, slope, width)
        return desc

    def get_adjacency_list(self, hallway_vertices):
        adj_list = {}
        obstacles = {}
        for e in hallway_vertices:
            first = e[0]
            second = e[1]
            if second in adj_list:
                adj_list[second].append(first)
            else:
                adj_list[second] = [first]
            if first in adj_list:
                adj_list[first].append(second)
            else:
                adj_list[first] = [second]

        return adj_list,obstacles

    def process_edges(self, edges):
        vs = []
        x_pairs = []
        y_pairs = []
        for edge in edges:
            xh, yh = self.reject_outliers(list(edge))
            if len(xh) == 2:
                x_pairs.append(xh)
                y_pairs.append(yh)
        return x_pairs, y_pairs

    def reject_outliers(self, data):
        raw_x = [v[0] for v in data]
        raw_y = [v[1] for v in data]
        rejected_points = [v for v in raw_x if v < 0]
        indexes = [i for i in range(len(raw_x)) if raw_x[i] in rejected_points]
        x_values = [raw_x[i] for i in range(len(raw_x)) if i not in indexes]
        y_values = [raw_y[i] for i in range(len(raw_y)) if i not in indexes]
        return x_values, y_values


"""
    Function for starting node and creating graph. 
"""
if __name__ == "__main__":
    rospy.init_node("create_graph")
    graph = Graph()
