#!/usr/bin/python
import copy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial import Voronoi
import time
import rospy
from threading import Lock
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from gvgexploration.msg import *
from gvgexploration.srv import *
import project_utils as pu
import tf
from time import sleep
import shapely.geometry as sg
from nav_msgs.msg import *
from nav2d_navigator.msg import *
from std_srvs.srv import *
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, save_data
from std_msgs.msg import String

SCALE = 10
FREE = 0.0
OCCUPIED = 100.0
UNKNOWN = -1.0
SMALL = 0.000000000001
FONT_SIZE = 16
MARKER_SIZE = 12


class Graph:
    def __init__(self, robot_id=-1):
        self.min_hallway_width = None
        self.height = 0
        self.width = 0
        self.pixel_desc = {}
        self.obstacles = {}
        self.free_points = {}
        self.resolution = 0.05
        self.plot_intersection_active = False
        self.plot_data_active = False
        self.lock = Lock()

        self.obstacles = {}
        self.adj_list = {}
        self.edges = {}
        self.leaf_slope = {}
        self.longest = None
        self.adj_dict = {}
        self.tree_size = {}
        self.leave_dict = {}
        self.parent_dict = {}
        self.new_information = {}
        self.known_points = {}
        self.unknown_points = {}
        self.performance_data = []
        self.last_intersection = None

        self.latest_map = None
        self.prev_ridge = None
        rospy.init_node("graph_node")
        self.robot_id = rospy.get_param("~robot_id")
        self.robot_count = rospy.get_param("~robot_count")
        self.environment = rospy.get_param("~environment")
        self.run = rospy.get_param("~run")
        self.debug_mode = rospy.get_param("~debug_mode")
        self.termination_metric = rospy.get_param("~termination_metric")
        self.min_hallway_width = rospy.get_param("~min_hallway_width".format(self.robot_id)) * pu.SCALE
        self.comm_range = rospy.get_param("~comm_range".format(self.robot_id)) * pu.SCALE
        self.point_precision = rospy.get_param("~point_precision".format(self.robot_id))
        self.min_edge_length = rospy.get_param("~min_edge_length".format(self.robot_id)) * pu.SCALE
        self.lidar_scan_radius = rospy.get_param("~lidar_scan_radius".format(self.robot_id)) * pu.SCALE
        self.lidar_fov = rospy.get_param("~lidar_fov".format(self.robot_id))
        self.slope_bias = rospy.get_param("~slope_bias".format(self.robot_id)) #* SCALE
        self.separation_bias = rospy.get_param("~separation_bias".format(self.robot_id)) #* pu.SCALE
        self.opposite_vector_bias = rospy.get_param("~opposite_vector_bias".format(self.robot_id))

        self.edge_pub = rospy.Publisher("/robot_{}/edge_list".format(self.robot_id), EdgeList, queue_size=10)
        rospy.Service('/robot_{}/frontier_points'.format(self.robot_id), FrontierPoint, self.frontier_point_handler)
        rospy.Service('/robot_{}/check_intersections'.format(self.robot_id), Intersections, self.intersection_handler)
        rospy.Service('/robot_{}/fetch_graph'.format(self.robot_id), FetchGraph, self.fetch_graph_handler)
        self.move_to_stop = rospy.ServiceProxy('/robot_{}/Stop'.format(self.robot_id), Trigger)
        rospy.Subscriber('/robot_{}/map'.format(self.robot_id), OccupancyGrid, self.map_callback)
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)
        self.already_shutdown = False
        self.robot_pose = None
        self.listener = tf.TransformListener()
        # edges
        self.deleted_nodes = {}
        self.deleted_obstacles = {}
        rospy.loginfo('Robot {}: Successfully created graph node'.format(self.robot_id))

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            try:
                r.sleep()
            except Exception as e:
                rospy.logerr('Robot {}: Graph node interrupted!: {}'.format(self.robot_id, e))

    def map_callback(self, data):
        self.latest_map = data

    def is_same_intersection(self, intersec, robot_pose):
        is_same = True
        if self.last_intersection:
            last_close_edge = self.last_intersection[0]
            current_close_edge = intersec[0]
            last_dist = min([pu.D(robot_pose, last_close_edge[0]), pu.D(robot_pose, last_close_edge[1])])
            current_dist = min([pu.D(robot_pose, current_close_edge[0]), pu.D(robot_pose, current_close_edge[1])])
            if abs(current_dist - last_dist) > self.comm_range:
                is_same = False
        return is_same

    def frontier_point_handler(self, request):
        self.lock.acquire()
        count = request.count
        map_msg = self.latest_map
        if not map_msg:
            map_msg = rospy.wait_for_message("/robot_{}/map".format(self.robot_id), OccupancyGrid)
        self.compute_graph(map_msg)
        start_time = rospy.Time.now().to_sec()
        self.compute_new_information()
        ppoints = []
        selected_leaves = []
        while len(self.new_information) > 0:
            best_edge = max(self.new_information, key=self.new_information.get)
            ridge = self.create_ridge(best_edge)
            selected_leaves.append(ridge)
            ppoints.append(best_edge[0][1])
            del self.new_information[best_edge]
            if len(selected_leaves) == count:
                break
        now = rospy.Time.now().to_sec()
        t = (now - start_time)
        self.performance_data.append(
            {'time': rospy.Time.now().to_sec(), 'type': 2, 'robot_id': self.robot_id, 'computational_time': t})
        if self.debug_mode:
            if not self.plot_data_active:
                self.plot_data(ppoints, is_initial=True)
        self.lock.release()
        return FrontierPointResponse(ridges=selected_leaves)

    def intersection_handler(self, data):
        self.lock.acquire()
        pose_data = data.pose
        map_msg = self.latest_map
        if not map_msg:
            map_msg = rospy.wait_for_message("/robot_{}/map".format(self.robot_id), OccupancyGrid)
        self.compute_graph(map_msg)
        robot_pose = [0.0] * 2
        robot_pose[INDEX_FOR_X] = pose_data.position.x
        robot_pose[INDEX_FOR_Y] = pose_data.position.y
        result = 0
        close_edge, intersecs = self.compute_intersections(robot_pose)
        if intersecs:
            intersec = intersecs[0]
            # if not self.is_same_intersection(intersec, robot_pose):
            self.last_intersection = intersec
            pu.log_msg(self.robot_id, "Intersection: {}".format(intersecs), self.debug_mode)
            result = pu.D(pu.scale_down(intersec[0][1]), pu.scale_down(intersec[1][0]))
        self.lock.release()
        return IntersectionsResponse(result=result)

    def fetch_graph_handler(self, data):
        self.lock.acquire()
        robot_pose = (data.pose.position.x, data.pose.position.y)
        map_msg = self.latest_map
        if not map_msg:
            map_msg = rospy.wait_for_message("/robot_{}/map".format(self.robot_id), OccupancyGrid)
        self.compute_graph(map_msg)
        edgelist = self.create_edge_list(robot_pose)
        self.lock.release()
        return FetchGraphResponse(edgelist=edgelist)

    def create_edge_list(self, robot_pose):
        alledges = list(self.edges)
        edgelist = EdgeList()
        edgelist.header.stamp = rospy.Time.now()
        edgelist.header.frame_id = str(self.robot_id)
        edgelist.ridges = []
        edgelist.pixels = []
        edge_dists = {}
        for e in alledges:
            if e in self.edges:
                obs = self.edges[e]
                ridge = self.create_ridge((e, obs))
                edgelist.ridges.append(ridge)
                if self.in_line_with_previous_edge(self.prev_ridge, (e, obs)):
                    d = pu.D(robot_pose, e[1])  # min([pu.D(robot_pose, e[0]), pu.D(robot_pose, e[1])])
                    edge_dists[(e, obs)] = d
        if edge_dists:
            new_close_ridge = min(edge_dists, key=edge_dists.get)
            # if self.prev_ridge and self.prev_ridge[0] == new_close_ridge[0]:
            #     new_close_ridge = ((self.prev_ridge[0][1], self.prev_ridge[0][0]), self.prev_ridge[1])
        else:
            new_close_ridge = ((self.prev_ridge[0][1], self.prev_ridge[0][0]), self.prev_ridge[1])
        new_closest_ridge = self.create_ridge(new_close_ridge)
        edgelist.close_ridge = new_closest_ridge
        self.prev_ridge = new_close_ridge
        for k, v in self.pixel_desc.items():
            if v != FREE:
                pix = Pixel()
                pix.pose.position.x = k[INDEX_FOR_X]
                pix.pose.position.y = k[INDEX_FOR_Y]
                pix.desc = v
                edgelist.pixels.append(pix)
        return edgelist

    def get_closest_edge(self, robot_pose, close_edges):
        linear_edge = {}
        chosen_edge = None
        for e in close_edges:
            if self.in_line_with_previous_edge(self.prev_ridge, e):
                d = pu.D(robot_pose, e[0][1])
                linear_edge[d] = e
        if linear_edge:
            chosen_edge = linear_edge[min(linear_edge.keys())]
        return chosen_edge

    def compute_graph(self, occ_grid):
        start_time = rospy.Time.now().to_sec()
        self.get_image_desc(occ_grid)
        try:
            self.compute_hallway_points()
            now = rospy.Time.now().to_sec()
            t = now - start_time
            self.performance_data.append(
                {'time': rospy.Time.now().to_sec(), 'type': 0, 'robot_id': self.robot_id, 'computational_time': t})
        except Exception as e:
            pu.log_msg(self.robot_id, 'Robot {}: Error in graph computation'.format(self.robot_id), self.debug_mode)

    def get_image_desc(self, occ_grid):
        resolution = occ_grid.info.resolution
        origin_pos = occ_grid.info.origin.position
        origin_x = origin_pos.x
        origin_y = origin_pos.y
        height = occ_grid.info.height
        width = occ_grid.info.width
        grid_values = np.array(occ_grid.data).reshape((height, width)).astype(np.float32)
        num_rows = grid_values.shape[0]
        num_cols = grid_values.shape[1]
        self.obstacles.clear()
        for row in range(num_rows):
            for col in range(num_cols):
                index = [0] * 2
                index[INDEX_FOR_Y] = num_rows - row - 1
                index[INDEX_FOR_X] = col
                index = tuple(index)
                pose = pu.pixel2pose(index, origin_x, origin_y, resolution)
                scaled_pose = pu.get_point(pu.scale_up(pose))
                p = grid_values[num_rows - row - 1, col]
                self.pixel_desc[scaled_pose] = p
                if p == OCCUPIED:
                    self.obstacles[scaled_pose] = OCCUPIED

    def compute_intersections(self, pose):
        start = rospy.Time.now().to_sec()
        intersecs = []
        close_edge = []
        robot_pose = pu.scale_up(pose)
        closest_ridge = {}
        edge_list = list(self.edges)
        vertex_dict = {}
        for e in edge_list:
            p1 = e[0]
            p2 = e[1]
            o = self.edges[e]
            width = pu.D(o[0], o[1])
            u_check = pu.W(p1, robot_pose) < width or pu.W(p2, robot_pose) < width
            if u_check:
                v1 = pu.get_vector(p1, p2)
                desc = (v1, width)
                vertex_dict[e] = desc
                d = min([pu.D(robot_pose, e[0]), pu.D(robot_pose, e[1])])
                closest_ridge[e] = d
        if closest_ridge:
            cr = min(closest_ridge, key=closest_ridge.get)
            if closest_ridge[cr] < vertex_dict[cr][1]:
                close_edge = cr
                intersecs = self.process_decision(vertex_dict, close_edge, robot_pose)
                if self.debug_mode:
                    if not self.plot_intersection_active and intersecs:
                        self.plot_intersections(None, close_edge, intersecs, robot_pose)

        now = rospy.Time.now().to_sec()
        t = (now - start)
        self.performance_data.append({'time': rospy.Time.now().to_sec(), 'type': 1, 'robot_id': self.robot_id, 'computational_time': t})
        return close_edge, intersecs

    def process_decision(self, vertex_descriptions, ridge, robot_pose):
        r_desc = vertex_descriptions[ridge]
        v1 = r_desc[0]
        w1 = r_desc[1]
        p1 = ridge[0]
        p2 = ridge[1]
        intesec_pose = {}
        for j, desc in vertex_descriptions.items():
            if j != ridge and (ridge[0] not in j or ridge[1] not in j):
                p3 = j[1]
                v2 = desc[0]
                w2 = desc[1]
                if self.lidar_scan_radius < pu.D(p2, p3) < self.comm_range:
                    if pu.collinear(p1, p2, p3, w1, self.slope_bias):
                        cos_theta, separation = pu.compute_similarity(v1, v2, ridge, j)
                        if -1 <= cos_theta <= -1 + self.opposite_vector_bias and abs(separation) < abs(w1 - w2):
                            intesec_pose[pu.D(p2, p3)] = (ridge, j)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass

        intersections = self.get_pairs_with_unknown_area(intesec_pose)
        # if intesec_pose:
        #     min_dist = min(intesec_pose.keys())
        #     intersections.append(intesec_pose[min_dist])
        return intersections

    def get_pairs_with_unknown_area(self, edge_pairs):
        edge_boxes = {}
        pair_dist = {}
        intersections = []
        for dist, pair in edge_pairs.items():
            p1 = pair[0][1]
            p2 = pair[1][1]
            x_min = min([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])
            y_min = min([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])
            x_max = max([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])
            y_max = max([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])
            min_points = max([abs(x_max - x_min), abs(y_max - y_min)])
            bbox = sg.box(x_min, y_min, x_max, y_max)
            edge_boxes[pair] = (bbox, min_points)
            pair_dist[pair] = dist
        if edge_boxes:
            actual_pairs = {}
            actual_intersecs = {}
            for p, v in self.pixel_desc.items():
                if v == UNKNOWN:
                    point = sg.Point(p[INDEX_FOR_X], p[INDEX_FOR_Y])
                    for pair, values in edge_boxes.items():
                        if values[0].contains(point):
                            if pair not in actual_intersecs:
                                actual_intersecs[pair] = 1
                            else:
                                actual_intersecs[pair] += 1
            for pair, count in actual_intersecs.items():
                if count >= edge_boxes[pair][1]:
                    actual_pairs[pair_dist[pair]] = pair
            if actual_pairs:
                min_dist = min(actual_pairs.keys())
                intersections.append(actual_pairs[min_dist])
        return intersections

    # def process_decision(self, ridge, robot_pose):
    #     p1 = ridge[0]
    #     p2 = ridge[1]
    #     obs1 = self.edges[ridge]
    #     parent = {p2: p1}
    #     visited = {}
    #     S = [p2]
    #     intesec_pose = {}
    #     intersections = []
    #     while len(S) > 0:
    #         u = S.pop()
    #         e = (parent[u], u)
    #         if e != ridge and (ridge[0] not in e or ridge[1] not in e):
    #             obs = self.edges[e]
    #             intersec = self.get_intersection(robot_pose, ridge, e, obs1, obs)
    #             if intersec:
    #                 intesec_pose[pu.D(robot_pose, u)] = intersec
    #         neighbors = self.adj_list[u]
    #         for v in neighbors:
    #             if v not in visited:
    #                 S.append(v)
    #                 parent[v] = u
    #         visited[u] = None
    #     if intesec_pose:
    #         min_dist = min(intesec_pose.keys())
    #         intersections.append(intesec_pose[min_dist])
    #     return intersections

    def get_intersection(self, robot_pose, e1, e2, o1, o2):
        p1 = e1[0]
        p2 = e1[1]
        v1 = pu.get_vector(p1, p2)
        width1 = pu.D(o1[0], o1[1])
        p3 = e2[0]
        p4 = e2[1]
        width2 = pu.D(o2[0], o2[1])
        v2 = pu.get_vector(p3, p4)
        intersec = None
        if self.lidar_scan_radius < pu.D(p2, p4) < self.comm_range:
            if pu.there_is_unknown_region(p2, p4, self.pixel_desc):
                if pu.collinear(p1, p2, p4, width1, self.slope_bias):
                    cos_theta, separation = pu.compute_similarity(v1, v2, e1, e2)
                    if -1 <= cos_theta <= -1 + self.opposite_vector_bias and abs(
                            separation - width1) < self.separation_bias:
                        intersec = (e1, e2)
                    else:
                        pass
                else:
                    pass
            else:
                pass
        return intersec

    def compute_hallway_points(self):
        # self.lock.acquire()
        obstacles = list(self.obstacles)
        if obstacles:
            vor = Voronoi(obstacles)
            vertices = vor.vertices
            ridge_vertices = vor.ridge_vertices
            ridge_points = vor.ridge_points
            # self.old_edges = copy.deepcopy(self.edges)
            # self.old_adj_list = copy.deepcopy(self.adj_list)
            self.edges.clear()
            for i in range(len(ridge_vertices)):
                ridge_vertex = ridge_vertices[i]
                ridge_point = ridge_points[i]
                p1 = [0] * 2
                p2 = [0] * 2
                p1[INDEX_FOR_X] = vertices[ridge_vertex[0]][INDEX_FOR_X]
                p1[INDEX_FOR_Y] = vertices[ridge_vertex[0]][INDEX_FOR_Y]
                p2[INDEX_FOR_X] = vertices[ridge_vertex[1]][INDEX_FOR_X]
                p2[INDEX_FOR_Y] = vertices[ridge_vertex[1]][INDEX_FOR_Y]
                p1 = tuple(p1)
                p2 = tuple(p2)
                e = (p1, p2)
                if self.is_free(p1) and self.is_free(
                        p2):  # and not self.already_exists(p1) and not self.already_exists(p2):  # self.points_within_range(p1, p2) and
                    q1 = obstacles[ridge_point[0]]
                    q2 = obstacles[ridge_point[1]]
                    o = (tuple(q1), tuple(q2))
                    if pu.D(q1, q2) > self.min_hallway_width:
                        self.edges[e] = o
            self.get_adjacency_list(self.edges)
            self.connect_subtrees()
            self.merge_similar_edges()
            # if self.old_adj_list and self.adj_list:
            #     self.merge_graphs()
        # self.lock.release()

    def merge_graphs(self):
        if self.old_edges and self.edges:
            edge1, subtree1, parents1 = self.localize_robot(self.robot_pose, self.old_adj_list, self.old_edges)
            edge2, subtree2, parents2 = self.localize_robot(self.robot_pose, self.adj_list, self.edges)
            edge_pair = (edge1, edge2)
            self.remove_subtrees(edge1, self.old_edges, self.old_adj_list)
            self.remove_subtrees(edge2, self.edges, self.adj_list)
            self.link_edges(edge_pair, self.old_edges, self.edges)
            self.get_adjacency_list(self.edges)
        else:
            print("empty subtree")

    def already_exists(self, p):
        pose = [0.0] * 2
        pose[INDEX_FOR_X] = round(p[INDEX_FOR_X])
        pose[INDEX_FOR_Y] = round(p[INDEX_FOR_Y])
        pose = tuple(pose)
        return pose in self.deleted_nodes

    def localize_robot(self, robot_pose, adj_list, edges):
        dists = {}
        s = list(adj_list.keys())[0]
        visited = {}
        parents = {s: None}
        S = [s]
        while len(S) > 0:
            u = S.pop()
            neighbors = adj_list[u]
            for v in neighbors:
                if v not in visited:
                    S.append(v)
                    parents[v] = u
                    dists[pu.D(robot_pose, v)] = (u, v)
            visited[u] = None
        close_edge = dists[min(dists.keys())]
        subtree, parents = self.get_subtree(close_edge, adj_list)
        return close_edge, subtree, parents

    def remove_subtrees(self, close_edge, edges, adj_list):
        visited = [close_edge[0]]
        S = [close_edge[1]]
        while len(S) > 0:
            u = S.pop()
            neighbors = adj_list[u]
            for v in neighbors:
                if v not in visited:
                    S.append(v)
                    e = (u, v)
                    if e != close_edge:
                        if e in edges:
                            del edges[e]
            visited.append(u)

    def link_edges(self, edge_pair, edges1, edges2):
        e1 = edge_pair[0]
        e2 = edge_pair[1]
        if e1 in edges1:
            obs1 = edges1[e1]
        else:
            obs1 = edges1[(e1[1], e1[0])]
        if e2 in edges2:
            obs2 = edges2[e2]
        else:
            obs2 = edges2[(e2[1], e2[0])]
        obs = obs1
        w1 = pu.D(obs1[0], obs1[1])
        w2 = pu.D(obs2[0], obs2[1])
        if w2 > w1:
            obs = obs2

        new_edge = (e1[1], e2[1])
        del edges1[e1]
        del edges2[e2]
        edges2[new_edge] = obs
        edges2.update(edges1)

    def get_subtree(self, edge, adj_list):
        S = [edge[1]]
        visited = [edge[0]]
        parents = {edge[1]: edge[0]}
        leaves = set()
        while len(S) > 0:
            u = S.pop()
            neighbors = adj_list[u]
            if len(neighbors) <= 1:
                leaves.add(u)
            for v in neighbors:
                if v not in visited:
                    S.append(v)
                    parents[v] = u
            visited.append(u)
        tree = set()
        for l in leaves:
            self.get_path(l, parents[l], parents, tree)
        subtree = list(tree)
        return subtree, parents

    def get_path(self, leaf, parent, parents, tree):
        l = leaf
        while parents[l] != parent:
            l = parents[l]
            tree.add(l)
        tree.add(l)

    def merge_records(self):
        old_nodes = list(self.adj_list)
        new_leaves = list(self.leaf_slope)
        dist_leaf = {}
        obstacle_dict = {}
        for n in old_nodes:
            neighbors = self.adj_list[n]
            obs = None
            for a in neighbors:
                if (n, a) in self.edges:
                    obs = self.edges[(n, a)]
                    break
            if obs:
                obstacle_dict[n] = obs
                for nl in new_leaves:
                    if nl in dist_leaf:
                        dist_leaf[nl][pu.D(nl, n)] = n
                    else:
                        dist_leaf[nl] = {pu.D(nl, n): n}
        neighbor_distance = {}
        obs_dict = {}
        for k, v in dist_leaf.items():
            dist = np.min(v.keys())
            neighbor_distance[(v[dist], k)] = dist
            obs_dict[(v[dist], k)] = obstacle_dict[v[dist]]
        if neighbor_distance:
            min_dist_key = min(neighbor_distance, key=neighbor_distance.get)
            self.edges[min_dist_key] = obs_dict[min_dist_key]
            self.get_adjacency_list(self.edges)

    def connect_subtrees(self, count=0):
        N = len(self.adj_list)
        self.get_deeepest_tree_source()
        if not N:
            return
        if N == len(self.adj_dict[self.longest]):  # or count == 10:
            self.get_adjacency_list(self.edges)
            return
        allnodes = []  # all leaves for every search
        for k, v in self.leave_dict.items():
            allnodes += v
        allnodes = list(set(allnodes))  # deal with only unique leaves
        for leaf, adj in self.adj_dict.items():
            adj_leaves = self.leave_dict[leaf]
            all_nodes = []
            # for lf, lf_adj in self.adj_dict.items():
            #     if lf != leaf:
            #         all_nodes += lf_adj.keys()
            # allnodes = list(set(all_nodes))  # deal with only unique leaves
            leaf_dist = {}
            for l in adj_leaves:
                dist_dict = {}
                for n2 in allnodes:
                    if n2 not in adj:
                        dist_dict[n2] = pu.D(l, n2)
                if dist_dict:
                    nei = min(dist_dict, key=dist_dict.get)
                    leaf_dist[l] = nei
            longest_dist = {k: pu.D(k, v) for k, v in leaf_dist.items()}
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
                        w1 = pu.D(cd[0], cd[1])
                        break
                for ad in cls_n:
                    clw2 = (closest, ad)
                    if (closest, ad) in self.edges:
                        cd1 = self.edges[(closest, ad)]

                        w2 = pu.D(cd1[0], cd1[1])
                        break
                if w1 > w2:
                    self.edges[(close_node, closest)] = clw1
                else:
                    self.edges[(close_node, closest)] = clw2
            else:
                self.get_adjacency_list(self.edges)
                return
        self.get_adjacency_list(self.edges)
        return self.connect_subtrees(count=count + 1)

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
                self.leaf_slope[k] = pu.theta(list(v)[0], k)

    def is_free(self, p):
        xc = int(np.round(p[INDEX_FOR_X]))
        yc = int(np.round(p[INDEX_FOR_Y]))
        new_p = [0] * 2
        new_p[INDEX_FOR_X] = xc
        new_p[INDEX_FOR_Y] = yc
        new_p = tuple(new_p)
        return new_p in self.pixel_desc and self.pixel_desc[new_p] == FREE

    def get_edge(self, ridge):
        edge = {}
        try:
            p1 = [0.0] * 2
            p1[INDEX_FOR_X] = ridge.nodes[0][INDEX_FOR_X]
            p1[INDEX_FOR_Y] = ridge.nodes[0][INDEX_FOR_Y]
            p1 = tuple(p1)
            p2 = [0.0] * 2
            p2[INDEX_FOR_X] = ridge.nodes[1][INDEX_FOR_X]
            p2[INDEX_FOR_Y] = ridge.nodes[1][INDEX_FOR_Y]
            p2 = tuple(p2)
            q1 = [0.0] * 2
            q1[INDEX_FOR_X] = ridge.obstacles[0][INDEX_FOR_X]
            q1[INDEX_FOR_Y] = ridge.obstacles[0][INDEX_FOR_Y]
            q1 = tuple(q1)
            q2 = [0.0] * 2
            q2[INDEX_FOR_X] = ridge.obstacles[1][INDEX_FOR_X]
            q2[INDEX_FOR_Y] = ridge.obstacles[1][INDEX_FOR_Y]
            q2 = tuple(q2)
            P = (p1, p2)
            Q = (q1, q2)
            edge[P] = Q
        except:
            pu.log_msg(self.robot_id, "Invalid goal", self.debug_mode)

        return edge

    def create_ridge(self, ridge):
        p1 = Pose()
        p1.position.x = ridge[0][0][INDEX_FOR_X]
        p1.position.y = ridge[0][0][INDEX_FOR_Y]

        p2 = Pose()
        p2.position.x = ridge[0][1][INDEX_FOR_X]
        p2.position.y = ridge[0][1][INDEX_FOR_Y]

        q1 = Pose()
        q1.position.x = ridge[1][0][INDEX_FOR_X]
        q1.position.y = ridge[1][0][INDEX_FOR_Y]

        q2 = Pose()
        q2.position.x = ridge[1][1][INDEX_FOR_X]
        q2.position.y = ridge[1][1][INDEX_FOR_Y]

        ridge = Ridge()
        ridge.nodes = [p1, p2]
        ridge.obstacles = [q1, q2]
        ridge.scale = pu.SCALE

        return ridge

    def in_line_with_previous_edge(self, prev_ridge, current_edge):
        in_line = False
        if not prev_ridge:
            in_line = True
        else:
            edge1 = prev_ridge[0]
            edge2 = current_edge[1]
            v1 = pu.get_vector(edge1[0], edge1[1])
            v2 = pu.get_vector(edge2[0], edge2[1])
            cos_theta, separation = pu.compute_similarity(v1, v2, edge1, edge2)
            if cos_theta >= 0:
                in_line = True
        return in_line

    def scatter_plot(self):
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
        obstacles = list(self.obstacles)

        xr = [v[INDEX_FOR_Y] for v in obstacles]
        yr = [v[INDEX_FOR_X] for v in obstacles]
        ax.scatter(xr, yr, color='black', marker="1")
        x_pairs, y_pairs = pu.process_edges(self.edges)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(y, x, "g-o")
        plt.grid()
        plt.axis('off')
        plt.savefig("gvg/map_update_{}_{}_{}.png".format(self.robot_id, time.time(), self.run))

        plt.close()
        # plt.show()

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
                all_neis = []
                if u in self.adj_list:
                    all_neis = self.adj_list[u]
                neighbors = [k for k in all_neis if k != parents[u]]
                if len(neighbors) == 1:
                    v = neighbors[0]
                    if v not in visited:
                        S.append(v)
                        if parents[u]:
                            us = pu.get_vector(parents[u], u)
                            ps = pu.get_vector(u, v)
                            cos_theta, separation = pu.compute_similarity(us, ps, (parents[u], u), (u, v))
                            if 1 - self.opposite_vector_bias <= cos_theta <= 1:
                                parents[v] = parents[u]
                                deleted_nodes[u] = None
                                self.adj_list[v].remove(u)
                                self.adj_list[v].add(parents[u])
                                self.adj_list[parents[u]].add(v)
                                if (u, parents[u]) in self.edges:
                                    self.edges[(parents[u], v)] = self.edges[(u, parents[u])]
                                else:
                                    self.edges[(parents[u], v)] = self.edges[(parents[u], u)]
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
                    if (k, l) in self.edges:
                        edges[(k, l)] = self.edges[(k, l)]
                    else:
                        edges[(k, l)] = self.edges[(l, k)]

        for k, v in new_adj_list.items():
            if len(v) == 1:
                self.leaf_slope[k] = pu.theta(k, list(v)[0])
        self.adj_list = new_adj_list
        self.edges = edges
        self.save_deleted_nodes(deleted_nodes)

    def save_deleted_nodes(self, deleted_nodes):
        for v, _ in deleted_nodes.items():
            pose = [0.0] * 2
            pose[INDEX_FOR_X] = round(v[INDEX_FOR_X])
            pose[INDEX_FOR_Y] = round(v[INDEX_FOR_Y])
            pose = tuple(pose)
            self.deleted_nodes[pose] = None  # add all the new deleted nodes

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

    def get_free_points(self):
        points = list(self.pixel_desc)
        return [p for p in points if self.pixel_desc[p] == FREE]

    def get_obstacles(self):
        points = list(self.pixel_desc)
        return [p for p in points if self.pixel_desc[p] == OCCUPIED]

    def compute_new_information(self):
        resolution = 1
        frontier_size = {}
        self.new_information.clear()
        self.known_points.clear()
        self.unknown_points.clear()
        leaf_copy = copy.deepcopy(self.leaf_slope)
        edges_copy = copy.deepcopy(self.edges)
        adjlist_copy = copy.deepcopy(self.adj_list)
        for leaf, slope in leaf_copy.items():
            obs = None
            edge = (list(adjlist_copy[leaf])[0], leaf)
            redge = (leaf, list(adjlist_copy[leaf])[0])
            if edge in edges_copy:
                obs = edges_copy[edge]
            elif redge in edges_copy:
                obs = edges_copy[redge]
            if obs:
                frontier_size[leaf] = pu.D(obs[0], obs[1])
                ks, us = self.area(leaf, slope)
                unknown_area = len(us) * resolution ** 2
                self.new_information[(edge, obs)] = unknown_area
                self.known_points[(edge, obs)] = ks
                self.unknown_points[(edge, obs)] = us

    def area(self, point, orientation):
        known_points = []
        unknown_points = []
        radius = int(round(self.lidar_scan_radius))
        for d in np.arange(0, radius, 1):
            distance_points = []
            for theta in range(-1 * self.lidar_fov // 2, self.lidar_fov + 1):
                angle = np.deg2rad(theta) + orientation
                x = point[INDEX_FOR_X] + d * np.cos(angle)
                y = point[INDEX_FOR_Y] + d * np.sin(angle)
                pt = [0.0] * 2
                pt[INDEX_FOR_X] = x
                pt[INDEX_FOR_Y] = y
                pt = tuple(pt)
                distance_points.append(pt)
            unique_points = list(set(distance_points))
            for p in unique_points:
                if pu.is_free(p, self.pixel_desc) or pu.is_obstacle(p, self.pixel_desc):
                    known_points.append(p)
                else:
                    unknown_points.append(p)
        return known_points, unknown_points

    def plot_data(self, frontiers, is_initial=False, vertext=None):
        self.plot_data_active = True
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
        try:
            if unknown_points:
                UX, UY = zip(*unknown_points)
                ax.scatter(UX, UY, marker='1', color='gray')
            obstacles = list(self.obstacles)
            xr, yr = zip(*obstacles)
            ax.scatter(xr, yr, color='black', marker="1")
            x_pairs, y_pairs = pu.process_edges(self.edges)
            for i in range(len(x_pairs)):
                x = x_pairs[i]
                y = y_pairs[i]
                ax.plot(x, y, "g-.")
            leaves = list(self.leaf_slope)
            lx, ly = zip(*leaves)
            ax.scatter(lx, ly, color='red', marker='*', s=MARKER_SIZE)

            fx, fy = zip(*frontiers)
            ax.scatter(fx, fy, color='goldenrod', marker="D", s=MARKER_SIZE + 4)
            pose = self.get_robot_pose()
            point = pu.scale_up([pose])
            ax.scatter(point[INDEX_FOR_X], point[INDEX_FOR_Y], color='goldenrod', marker="D", s=MARKER_SIZE + 4)
            plt.grid()
        except:
            pass
        # plt.axis('off')
        plt.savefig("gvg/plot_{}_{}_{}.png".format(self.robot_id, time.time(), self.run))
        plt.close()
        # plt.show()
        self.plot_data_active = False

    def plot_intersections(self, ax, ridge, intersections, point):
        self.plot_intersection_active = True
        obstacles = list(self.obstacles)
        fig, ax = plt.subplots(figsize=(16, 10))

        ox = [v[INDEX_FOR_X] for v in obstacles]
        oy = [v[INDEX_FOR_Y] for v in obstacles]
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
            ax.plot(target_x[i], target_y[i], "r-d")

        for i in range(len(current_x)):
            ax.plot(current_x[i], current_y[i], "m-d")

        ax.scatter(point[INDEX_FOR_X], point[INDEX_FOR_Y], marker='*', color='purple')
        ax.scatter(ox, oy, color='black', marker='s')

        x_pairs, y_pairs = pu.process_edges(self.edges)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "g-.")
        plt.grid()
        plt.savefig(
            "gvg/intersections_{}_{}_{}.png".format(self.robot_id, time.time(),
                                                    self.run))  # TODO consistent time.
        plt.close(fig)
        self.plot_intersection_active = False

    def get_robot_pose(self):
        robot_pose = None
        while not robot_pose:
            try:
                self.listener.waitForTransform("robot_{}/map".format(self.robot_id),
                                               "robot_{}/base_link".format(self.robot_id), rospy.Time(),
                                               rospy.Duration(4.0))
                (robot_loc_val, rot) = self.listener.lookupTransform("robot_{}/map".format(self.robot_id),
                                                                     "robot_{}/base_link".format(self.robot_id),
                                                                     rospy.Time(0))
                robot_pose = (math.floor(robot_loc_val[0]), math.floor(robot_loc_val[1]), robot_loc_val[2])
                sleep(1)
            except:
                pass

        return robot_pose

    def shutdown_callback(self, msg):
        self.save_all_data()
        rospy.signal_shutdown('Graph: Shutdown command received!')

    def save_all_data(self):
        save_data(self.performance_data,
                  "gvg/performance_{}_{}_{}_{}_{}.pickle".format(self.environment, self.robot_count, self.run,
                                                                 self.termination_metric, self.robot_id))


if __name__ == "__main__":
    graph = Graph()
    graph.spin()
