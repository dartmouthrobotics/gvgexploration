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
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose
from gvgexploration.msg import *
from gvgexploration.srv import *
from numpy.linalg import norm
import project_utils as pu
import tf
from time import sleep
from nav_msgs.msg import *
from nav2d_navigator.msg import *
from std_srvs.srv import *
from actionlib_msgs.msg import GoalStatusArray, GoalID
from project_utils import INDEX_FOR_X, INDEX_FOR_Y

SCALE = 10
FREE = 0.0
OCCUPIED = 100.0
UNKNOWN = -1.0
SMALL = 0.000000000001
FONT_SIZE = 16
MARKER_SIZE = 12

ACTIVE_STATE = 1  # This state shows that the robot is collecting messages
PASSIVE_STATE = -1  # This state shows  that the robot is NOT collecting messages
ACTIVE = 1  # The goal is currently being processed by the action server
SUCCEEDED = 3  # The goal was achieved successfully by the action server (Terminal State)
ABORTED = 4  # The goal was aborted during execution by the action server due to some failure (Terminal State)
LOST = 9  # An action client can determine that a goal is LOST. This should not be sent over the wire by an action


class Graph:
    def __init__(self, robot_id=-1):
        self.min_hallway_width = None
        self.height = 0
        self.width = 0
        self.pixel_desc = {}
        self.obstacles = {}
        self.free_points = {}
        self.resolution = 0.05
        self.origin_x = 0
        self.origin_y = 0
        self.grid_values = np.zeros((1, 1))
        self.is_active = False
        self.plot_intersection_active = False
        self.plot_data_acive = False
        self.lock = Lock()

        self.obstacles = {}
        self.adj_list = {}
        self.leaves = {}
        self.edges = {}
        self.deleted_nodes = {}
        self.old_edges = {}
        self.old_leaf_slope = {}
        self.old_obstacles = {}
        self.old_adj_list = {}
        self.processed_obstacles = {}
        self.leaf_slope = {}
        self.longest = None
        self.adj_dict = {}
        self.tree_size = {}
        self.leave_dict = {}
        self.parent_dict = {}
        self.vertex_dict = {}
        self.update_active = False
        self.new_information = {}
        self.known_points = {}
        self.unknown_points = {}
        self.robot_count = 0
        self.environment = 'roscbt'
        self.waiting_for_plan = False
        self.navigation_plan = None
        self.move_failed = False
        self.covered_area = {}
        self.total_area = 0
        self.test_count = 0
        rospy.init_node("graph_node")
        self.last_map_update_time = rospy.Time.now().secs
        self.robot_id = rospy.get_param("~robot_id")
        self.robot_count = rospy.get_param("~robot_count")
        self.environment = rospy.get_param("~environment")
        self.run = rospy.get_param("~run")
        self.edge_pub = rospy.Publisher("/robot_{}/edge_list".format(self.robot_id), EdgeList, queue_size=10)
        self.min_hallway_width = rospy.get_param("/robot_{}/min_hallway_width".format(self.robot_id)) * pu.SCALE
        self.comm_range = rospy.get_param("/robot_{}/comm_range".format(self.robot_id)) * pu.SCALE
        self.point_precision = rospy.get_param("/robot_{}/point_precision".format(self.robot_id))
        self.min_edge_length = rospy.get_param("/robot_{}/min_edge_length".format(self.robot_id)) * pu.SCALE
        self.lidar_scan_radius = rospy.get_param("/robot_{}/lidar_scan_radius".format(self.robot_id)) * pu.SCALE
        self.info_scan_radius = rospy.get_param("/robot_{}/info_scan_radius".format(self.robot_id)) * pu.SCALE
        self.lidar_fov = rospy.get_param("/robot_{}/lidar_fov".format(self.robot_id))
        self.slope_bias = rospy.get_param("/robot_{}/slope_bias".format(self.robot_id)) * SCALE
        self.separation_bias = rospy.get_param("/robot_{}/separation_bias".format(self.robot_id)) * pu.SCALE
        self.opposite_vector_bias = rospy.get_param("/robot_{}/opposite_vector_bias".format(self.robot_id))
        rospy.Subscriber("/robot_{}/map".format(self.robot_id), OccupancyGrid, self.map_update_callback, queue_size=1)
        rospy.Service('/robot_{}/frontier_points'.format(self.robot_id), FrontierPoint, self.frontier_point_handler)
        rospy.Service('/robot_{}/check_intersections'.format(self.robot_id), Intersections, self.intersection_handler)
        rospy.Subscriber("/robot_{}/navigator/plan".format(self.robot_id), GridCells, self.navigation_plan_callback)
        self.moveTo_pub = rospy.Publisher("/robot_{}/MoveTo/goal".format(self.robot_id), MoveToPosition2DActionGoal,
                                          queue_size=10)
        rospy.Subscriber("/robot_{}/MoveTo/result".format(self.robot_id), MoveToPosition2DActionResult,
                         self.move_result_callback)
        self.coverage_pub = rospy.Publisher('/robot_{}/coverage'.format(self.robot_id), Coverage, queue_size=10)
        self.move_to_stop = rospy.ServiceProxy('/robot_{}/Stop'.format(self.robot_id), Trigger)
        self.robot_pose = None
        self.listener = tf.TransformListener()
        self.spin()

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            try:
                self.publish_edges()
                self.publish_coverage()
                r.sleep()
            except Exception as e:
                rospy.logerr('Robot {}: Graph node interrupted!: {}'.format(self.robot_id, e))
                break

    def publish_coverage(self):
        cov_msg = Coverage()
        cov_msg.header.stamp = rospy.Time.now()
        cov_ratio = self.get_coverage_ratio()
        cov_msg.coverage = cov_ratio
        self.coverage_pub.publish(cov_msg)

    def map_update_callback(self, occ_grid):
        self.last_map_update_time = rospy.Time.now().secs
        if not self.update_active:
            self.update_active = True
            start_time = time.time()
            self.resolution = occ_grid.info.resolution
            origin_pos = occ_grid.info.origin.position
            self.origin_x = origin_pos.x
            self.origin_y = origin_pos.y
            self.height = occ_grid.info.height
            self.width = occ_grid.info.width
            self.grid_values = np.array(occ_grid.data).reshape((self.height, self.width)).astype(np.float32)
            self.get_image_desc()
            try:
                self.compute_hallway_points()
                self.old_leaf_slope = copy.deepcopy(self.leaf_slope)
                self.old_edges = copy.deepcopy(self.edges)
                self.old_adj_list = copy.deepcopy(self.adj_list)
                now = time.time()
                t = now - start_time
                self.save_data([{'time': now, 'type': 0, 'robot_id': self.robot_id, 'computational_time': t}],
                               "gvg/performance_{}_{}_{}.pickle".format(self.environment, self.robot_count, self.run))
            except Exception as e:
                rospy.logerr('Robot {}: Error in graph computation: {}'.format(self.robot_id, e.message))
            self.update_active = False

    def frontier_point_handler(self, request):
        count = request.count
        rospy.logerr("Robot Count received: {}".format(count))
        start_time = time.time()
        self.compute_new_information()
        ppoints = []
        selected_poses = []
        while len(self.new_information) > 0:
            best_point = max(self.new_information, key=self.new_information.get)
            new_p = pu.scale_down(best_point)
            pose = Pose()
            pose.position.x = new_p[INDEX_FOR_X]
            pose.position.y = new_p[INDEX_FOR_Y]
            selected_poses.append(pose)
            ppoints.append(best_point)
            del self.new_information[best_point]
            if len(selected_poses) == count:
                break
        now = time.time()
        t = (now - start_time)
        self.save_data([{'time': now, 'type': 2, 'robot_id': self.robot_id, 'computational_time': t}],
                       "gvg/performance_{}_{}_{}.pickle".format(self.environment, self.robot_count, self.run))
        if not self.plot_data_acive:
            self.plot_data(ppoints, is_initial=True)
        return FrontierPointResponse(poses=selected_poses)

    def intersection_handler(self, data):
        pose_data = data.poses
        robot_poses = []
        for p in pose_data:
            robot_poses.append((p.position.x, p.position.y, p.position.z))
        result = 0
        close_edge, intersecs = self.compute_intersections(robot_poses)
        if intersecs:
            result = 1
        return IntersectionsResponse(result=result)

    def compute_intersections(self, robot_poses):
        start = time.time()
        intersecs = []
        close_edge = []
        robot_pose = self.get_robot_pose()
        robot_pose = pu.scale_up(robot_pose)  # pu.pose2pixel(robot_pose, self.origin_x, self.origin_y, self.resolution)
        closest_ridge = {}
        edge_list = list(self.old_edges)
        vertex_dict = {}
        for e in edge_list:
            p1 = e[0]
            p2 = e[1]
            o = self.old_edges[e]
            width = pu.D(o[0], o[1])
            if pu.D(p1, p2) > self.min_edge_length:
                u_check = pu.W(p1, robot_pose) < width or pu.W(p2, robot_pose) < width
                if u_check:
                    v1 = pu.get_vector(p1, p2)
                    desc = (v1, width)
                    vertex_dict[e] = desc
                    # d = norm(np.cross(v1, self.get_vector(robot_pose, p1))) / norm(v1)
                    d = min([pu.D(robot_pose, e[0]), pu.D(robot_pose, e[1])])
                    closest_ridge[e] = d
        if closest_ridge:
            cr = min(closest_ridge, key=closest_ridge.get)
            if closest_ridge[cr] < vertex_dict[cr][1]:
                close_edge = cr
                intersecs = self.process_decision(vertex_dict, close_edge, robot_pose)
                if not self.plot_intersection_active and intersecs:
                    self.plot_intersections(None, close_edge, intersecs, robot_pose, robot_poses)

                now = time.time()
                t = (now - start)
                self.save_data([{'time': now, 'type': 1, 'robot_id': self.robot_id, 'computational_time': t}],
                               "gvg/performance_{}_{}_{}.pickle".format(self.environment, self.robot_count, self.run))

        return close_edge, intersecs

    def process_decision(self, vertex_descriptions, ridge, robot_pose):
        r_desc = vertex_descriptions[ridge]
        v1 = r_desc[0]
        w1 = r_desc[1]
        p1 = ridge[0]
        p2 = ridge[1]
        intersections = []
        intesec_pose = {}
        for j, desc in vertex_descriptions.items():
            if j != ridge and (ridge[0] not in j or ridge[1] not in j):
                p3 = j[1]
                v2 = desc[0]
                if self.lidar_scan_radius < pu.D(robot_pose, p3) < self.comm_range:
                    if pu.there_is_unknown_region(p2, p3, self.pixel_desc):
                        if pu.collinear(p1, p2, p3, w1, self.slope_bias):
                            cos_theta, separation = pu.compute_similarity(v1, v2, ridge, j)
                            if -1 <= cos_theta <= -1 + self.opposite_vector_bias and abs(
                                    separation - w1) < self.separation_bias:
                                # intersections.append((ridge, j))
                                intesec_pose[pu.D(robot_pose, p3)] = (ridge, j)
                            else:
                                rospy.logerr("Cos theta: {}".format(cos_theta))
                                pass
                        else:
                            # rospy.logerr("Collinear: {}".format(pu.collinear(p1, p2, p3, w1, self.slope_bias)))
                            pass
                    else:
                        # rospy.logerr("Is in Unknown region: {}".format(pu.there_is_unknown_region(p2, p3, self.pixel_desc)))
                        pass
                else:
                    # rospy.logerr('Distance: {}'.format(pu.D(robot_pose, p3)))
                    pass
        if intesec_pose:
            min_dist = min(intesec_pose.keys())
            intersections.append(intesec_pose[min_dist])
        return intersections

    def check_point(self, point):
        self.waiting_for_plan = True
        self.navigation_plan = None
        self.move_failed = False
        self.test_move_to(point)
        start_time = rospy.Time.now().secs
        while not self.navigation_plan and not self.move_failed:
            if (rospy.Time.now().secs - start_time) > 10:  # only wait for 10 secs
                self.move_failed = True
            continue
        self.waiting_for_plan = False
        if self.navigation_plan:
            self.move_to_stop()
            return True
        elif self.move_failed:
            try:
                self.move_to_stop()
            except:
                pass
            self.move_failed = False
            return False
        return False

    def compute_hallway_points(self):
        obstacles = self.get_frontier_obstacles()
        if obstacles:
            vor = Voronoi(obstacles)
            vertices = vor.vertices
            ridge_vertices = vor.ridge_vertices
            ridge_points = vor.ridge_points
            self.edges.clear()
            for i in range(len(ridge_vertices)):
                ridge_vertex = ridge_vertices[i]
                ridge_point = ridge_points[i]
                if ridge_vertex[0] != -1 or ridge_vertex[1] != -1:
                    p1 = [0.0] * 2
                    p2 = [0.0] * 2
                    p1[INDEX_FOR_X] = vertices[ridge_vertex[0]][INDEX_FOR_X]
                    p1[INDEX_FOR_Y] = vertices[ridge_vertex[0]][INDEX_FOR_Y]
                    p2[INDEX_FOR_X] = vertices[ridge_vertex[1]][INDEX_FOR_X]
                    p2[INDEX_FOR_Y] = vertices[ridge_vertex[1]][INDEX_FOR_Y]
                    q1 = obstacles[ridge_point[0]]
                    q2 = obstacles[ridge_point[1]]
                    o = (q1, q2)
                    p1 = pu.get_point(p1)
                    p2 = pu.get_point(p2)
                    if pu.is_free(p1, self.pixel_desc) and pu.is_free(p2, self.pixel_desc) and pu.D(q1,
                                                                                                    q2) > self.min_hallway_width:
                        e = (p1, p2)
                        self.edges[e] = o
            self.get_adjacency_list(self.edges)
            self.connect_subtrees()
            self.merge_similar_edges()

    def wait_for_map_update(self):
        r = rospy.Rate(0.1)
        while not self.is_time_to_moveon():
            r.sleep()

    def is_time_to_moveon(self):
        diff = rospy.Time.now().secs - self.last_map_update_time
        return diff > 10

    def navigation_plan_callback(self, data):
        if self.waiting_for_plan:
            self.navigation_plan = data.cells
            self.move_to_stop()

    def move_result_callback(self, data):
        if data.status:
            if data.status.status == ABORTED:
                self.move_failed = True

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

    def test_move_to(self, goal):
        move = MoveToPosition2DActionGoal()
        move.header.frame_id = '/robot_{}/map'.format(self.robot_id)
        goal_id = GoalID()
        goal_id.id = 'robot_{}_move_test_{}'.format(self.robot_id, self.test_count)
        move.goal_id = goal_id
        move.goal.target_pose.x = goal[0]
        move.goal.target_pose.y = goal[1]
        self.moveTo_pub.publish(move)
        self.test_count += 1

    def publish_edges(self):
        self.lock.acquire()
        alledges = list(self.old_edges)
        edgelist = EdgeList()
        edgelist.header.stamp = rospy.Time.now()
        edgelist.header.frame_id = str(self.robot_id)
        edgelist.width = self.width
        edgelist.height = self.height
        edgelist.origin_x = self.origin_x
        edgelist.origin_y = self.origin_y
        edgelist.resolution = self.resolution
        edgelist.pixels = []
        edgelist.edges = []
        for r in alledges:
            ridge = Ridge()
            obs = self.old_edges[r]
            ridge.px = []
            ridge.py = []
            ridge.px = [r[0][INDEX_FOR_X], r[1][INDEX_FOR_X], obs[0][INDEX_FOR_X], obs[1][INDEX_FOR_X]]
            ridge.py = [r[0][INDEX_FOR_Y], r[1][INDEX_FOR_Y], obs[0][INDEX_FOR_Y], obs[1][INDEX_FOR_Y]]
            edgelist.edges.append(ridge)
        self.edge_pub.publish(edgelist)
        self.lock.release()

    def connect_subtrees(self, count=0):
        N = len(self.adj_list)
        self.get_deeepest_tree_source()
        if not N:
            return
        if N == len(self.adj_dict[self.longest]):  # or count == 10:
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
        obstacles = list(self.old_obstacles)

        xr = [v[INDEX_FOR_Y] for v in obstacles]
        yr = [v[INDEX_FOR_X] for v in obstacles]
        ax.scatter(xr, yr, color='black', marker="1")
        x_pairs, y_pairs = pu.process_edges(self.old_edges)
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
        self.deleted_nodes.update(deleted_nodes)

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
        self.lock.acquire()
        resolution = 1
        frontier_size = {}
        self.new_information.clear()
        self.known_points.clear()
        self.unknown_points.clear()
        leaf_copy = copy.deepcopy(self.leaf_slope)
        edges_copy = copy.deepcopy(self.edges)
        adjlist_copy=copy.deepcopy(self.adj_list)
        for leaf, slope in leaf_copy.items():
            # new_p = pu.scale_down(leaf)
            # is_good_point = self.check_point(new_p)
            # if is_good_point:
            obs = None
            if (list(adjlist_copy[leaf])[0], leaf) in edges_copy:
                obs = edges_copy[(list(adjlist_copy[leaf])[0], leaf)]
            elif (leaf, list(adjlist_copy[leaf])[0]) in edges_copy:
                obs = edges_copy[(leaf, list(adjlist_copy[leaf])[0])]
            if obs:
                frontier_size[leaf] = pu.D(obs[0], obs[1])
                ks, us = self.area(leaf, slope)
                unknown_area = len(us) * resolution ** 2
                # if leaf not in self.new_information:
                self.new_information[leaf] = unknown_area
                self.known_points[leaf] = ks
                self.unknown_points[leaf] = us
            # else:
            #     rospy.logerr('Point unreachable: {}'.format(new_p))
        self.lock.release()

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
                if self.origin_x * SCALE < x and self.origin_y * SCALE < y:
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
        self.plot_data_acive = True
        self.lock.acquire()
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
            obstacles = list(self.old_obstacles)
            xr, yr = zip(*obstacles)
            ax.scatter(xr, yr, color='black', marker="1")
            x_pairs, y_pairs = pu.process_edges(self.old_edges)
            for i in range(len(x_pairs)):
                x = x_pairs[i]
                y = y_pairs[i]
                ax.plot(x, y, "g-.")
            leaves = list(self.old_leaf_slope)
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
        self.plot_data_acive = False
        self.lock.release()

    def plot_intersections(self, ax, ridge, intersections, point, robot_poses):
        self.plot_intersection_active = True
        self.lock.acquire()
        robot_x = []
        robot_y = []
        for r in robot_poses:
            pixel = pu.scale_up(r)  # pu.pose2pixel(r, self.origin_x, self.origin_y, self.resolution)
            robot_x.append(pixel[INDEX_FOR_X])
            robot_y.append(pixel[INDEX_FOR_Y])

        obstacles = list(self.old_obstacles)
        # unknowns = list(self.unknowns)
        # if not ax:
        fig, ax = plt.subplots(figsize=(16, 10))

        # ux = [v[INDEX_FOR_X] for v in unknowns]
        # uy = [v[INDEX_FOR_Y] for v in unknowns]

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
        ax.scatter(robot_x, robot_y, marker='s', color='purple')
        ax.scatter(ox, oy, color='black', marker='s')
        # ax.scatter(ux, uy, color='gray', marker='1')

        x_pairs, y_pairs = pu.process_edges(self.old_edges)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "g-.")
        # plt.axis('off')
        plt.grid()
        plt.savefig(
            "gvg/intersections_{}_{}_{}.png".format(self.robot_id, time.time(), self.run))  # TODO consistent time.
        plt.close(fig)
        # plt.show()
        self.plot_intersection_active = False
        self.lock.release()

    def get_image_desc(self):
        self.lock.acquire()
        num_rows = self.grid_values.shape[0]
        num_cols = self.grid_values.shape[1]
        self.obstacles.clear()
        covered_area = {}
        for row in range(num_rows):
            for col in range(num_cols):
                index = [0] * 2
                index[INDEX_FOR_Y] = num_rows - row - 1
                index[INDEX_FOR_X] = col
                index = tuple(index)
                pose = pu.pixel2pose(index, self.origin_x, self.origin_y, self.resolution)
                scaled_pose = pu.get_point(pu.scale_up(pose))
                p = self.grid_values[num_rows - row - 1, col]
                self.pixel_desc[scaled_pose] = p
                if p == FREE:
                    covered_area[scaled_pose] = None
                elif p == OCCUPIED:
                    self.obstacles[scaled_pose] = OCCUPIED
                    covered_area[scaled_pose] = None
        self.covered_area.update(covered_area)
        if self.obstacles:
            self.old_obstacles = copy.deepcopy(self.obstacles)
        self.lock.release()

    def get_frontier_obstacles(self):
        self.lock.acquire()
        # all_points = list(self.pixel_desc)
        # obs_set = set()
        # for v in all_points:
        #     if self.pixel_desc[v] == OCCUPIED:
        #         obs_set.add(v)
        obstacles = list(self.old_obstacles)
        self.lock.release()
        return obstacles

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

    def get_coverage_ratio(self):
        ratio = 0.0
        new_total_area = float(len(self.pixel_desc))
        if new_total_area:
            ratio = len(self.covered_area) / new_total_area
        return ratio

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


if __name__ == "__main__":
    graph = Graph()
    filename = "map_message1.pickle"
    # graph.load_map_data(filename)
