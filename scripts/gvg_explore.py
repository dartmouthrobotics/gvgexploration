#!/usr/bin/python
from threading import Thread

import matplotlib
import uuid
import shapely.geometry as sg
from shapely.geometry.polygon import Polygon

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import project_utils as pu
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, SCALE
import actionlib
import rospy
import math
from gvgexploration.msg import EdgeList, Coverage, Ridge
from nav2d_navigator.msg import MoveToPosition2DActionGoal, MoveToPosition2DActionResult
from actionlib_msgs.msg import GoalStatusArray, GoalID
from std_srvs.srv import Trigger
from nav_msgs.msg import GridCells, Odometry
from geometry_msgs.msg import Twist, Pose, Point
import time
from time import sleep
import tf
from collections import deque
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from gvgexploration.srv import *

FREE = 0.0
OCCUPIED = 100.0
UNKNOWN = -1.0
MAX_COVERAGE_RATIO = 0.8
INF = 1000000000000
NEG_INF = -1000000000000
MAX_ATTEMPTS = 2
RR_TYPE = 2
ROBOT_SPACE = 1
TO_RENDEZVOUS = 1
TO_FRONTIER = 0
FROM_EXPLORATION = 4


class GVGExplore:
    def __init__(self):
        self.has_arrived = False
        self.navigation_failed = False
        self.received_first = False
        self.navigation_plan = None
        self.waiting_for_plan = True
        self.goal_count = 0
        self.edges = {}
        self.leaves = {}
        self.adj_list = {}
        self.pixel_desc = {}
        self.traveled_distance = []
        self.prev_explored = None
        self.map_width = 0
        self.map_height = 0
        self.robot_pose = None
        # self.received_choices = {}
        self.failed_points = []
        self.move_attempt = 0
        self.coverage_ratio = 0.0
        self.current_point = None
        self.robot_ranges = []
        self.publisher_map = {}
        self.close_devices = []
        self.explore_computation = []
        self.is_exploring = False
        self.moving_to_frontier = False
        self.is_processing_graph = False
        self.cancel_request = False
        self.motion_failed = False
        self.start_time = 0
        self.prev_pose = 0
        self.updated_graph = None
        rospy.init_node("gvgexplore", anonymous=True)
        self.robot_id = rospy.get_param('~robot_id')
        self.run = rospy.get_param("~run")
        self.debug_mode = rospy.get_param("~debug_mode")
        self.termination_metric = rospy.get_param("~termination_metric")
        self.graph_scale = rospy.get_param('~graph_scale')
        self.frontier_threshold = rospy.get_param('~frontier_threshold')
        self.min_edge_length = rospy.get_param("~min_edge_length".format(self.robot_id)) * self.graph_scale
        self.min_hallway_width = rospy.get_param("~min_hallway_width".format(self.robot_id)) * self.graph_scale
        self.comm_range = rospy.get_param("~comm_range".format(self.robot_id)) * self.graph_scale
        self.point_precision = rospy.get_param("~point_precision".format(self.robot_id))
        self.lidar_scan_radius = rospy.get_param("~lidar_scan_radius".format(self.robot_id)) * self.graph_scale
        self.lidar_fov = rospy.get_param("~lidar_fov")
        self.slope_bias = rospy.get_param("~slope_bias") * self.graph_scale
        self.separation_bias = rospy.get_param("~separation_bias".format(self.robot_id)) * self.graph_scale
        self.opposite_vector_bias = rospy.get_param("~opposite_vector_bias") * self.graph_scale
        self.target_distance = rospy.get_param('~target_distance')
        self.target_angle = rospy.get_param('~target_angle')
        self.robot_count = rospy.get_param("~robot_count")
        self.environment = rospy.get_param("~environment")
        self.max_coverage_ratio = rospy.get_param("~max_coverage")
        self.method = rospy.get_param("~method")
        rospy.Subscriber("/robot_{}/MoveTo/status".format(self.robot_id), GoalStatusArray, self.move_status_callback)
        rospy.Subscriber("/robot_{}/MoveTo/result".format(self.robot_id), MoveToPosition2DActionResult,
                         self.move_result_callback)
        rospy.Subscriber("/robot_{}/navigator/plan".format(self.robot_id), GridCells, self.navigation_plan_callback)
        self.move_to_stop = rospy.ServiceProxy('/robot_{}/Stop'.format(self.robot_id), Trigger)
        self.moveTo_pub = rospy.Publisher("/robot_{}/MoveTo/goal".format(self.robot_id), MoveToPosition2DActionGoal,
                                          queue_size=10)
        self.vertex_publisher = rospy.Publisher("/robot_{}/explore/vertices".format(self.robot_id), Marker,
                                                queue_size=10)

        self.leaf_publisher = rospy.Publisher("/robot_{}/explore/leaf_region".format(self.robot_id), Marker,
                                              queue_size=10)
        self.edge_publisher = rospy.Publisher("/robot_{}/explore/edges".format(self.robot_id), Marker, queue_size=10)
        self.fetch_graph = rospy.ServiceProxy('/robot_{}/fetch_graph'.format(self.robot_id), FetchGraph)
        self.pose_publisher = rospy.Publisher("/robot_{}/cmd_vel".format(self.robot_id), Twist, queue_size=1)
        rospy.Subscriber("/odom".format(self.robot_id), Odometry, callback=self.pose_callback)
        rospy.Subscriber('/robot_{}/gvgexplore/goal'.format(self.robot_id), Ridge, self.initial_action_handler)
        rospy.Service('/robot_{}/gvgexplore/cancel'.format(self.robot_id), CancelExploration,
                      self.received_prempt_handler)
        self.goal_feedback_pub = rospy.Publisher("/robot_{}/gvgexplore/feedback".format(self.robot_id), Pose,
                                                 queue_size=1)

        self.listener = tf.TransformListener()
        rospy.on_shutdown(self.save_all_data)
        self.already_shutdown = False
        rospy.loginfo("Robot {}: Exploration server online...".format(self.robot_id))

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            r.sleep()

    def process_graph(self):
        pixels = self.updated_graph.pixels
        ridges = self.updated_graph.ridges
        self.edges.clear()
        self.pixel_desc.clear()
        for r in ridges:
            edge = self.get_edge(r)
            self.edges.update(edge)
        for p in pixels:
            point = [0.0] * 2
            point[INDEX_FOR_X] = round(p.pose.position.x, 2)
            point[INDEX_FOR_Y] = round(p.pose.position.y, 2)
            point = tuple(point)
            self.pixel_desc[point] = p.desc
        self.create_adjlist()
        close_ridge = self.get_edge(self.updated_graph.close_ridge)
        edge = close_ridge.keys()[0]
        return edge

    def received_prempt_handler(self, data):
        rospy.logerr("Robot {}: GVGExplore action preempted".format(self.robot_id))
        thread = Thread(target=self.stop_navigator, args=())
        thread.start()
        return CancelExplorationResponse(result=1)

    def stop_navigator(self):
        self.cancel_request = True
        if self.moving_to_frontier:
            self.move_to_stop()
            self.moving_to_frontier = False
            while self.cancel_request:
                sleep(1)

    def initial_action_handler(self, ridge):
        rospy.logerr("Robot {}: GVGExplore received new goal".format(self.robot_id))
        ridge_dict = self.get_edge(ridge)
        edge = ridge_dict.keys()[0]
        leaf = edge[1]
        scaled_pose = pu.scale_down(leaf, self.graph_scale)
        self.moving_to_frontier = True
        self.motion_failed = False
        self.move_robot_to_goal(scaled_pose, 0)
        while self.moving_to_frontier:
            sleep(1)
        p = self.get_robot_pose()
        pose = Pose()
        pose.position.x = p[INDEX_FOR_X]
        pose.position.y = p[INDEX_FOR_Y]
        self.goal_feedback_pub.publish(pose)  # publish to feedback
        self.start_gvg_exploration(edge)

    # def start_gvg_exploration(self, edge):
    #     parent_id = self.get_id()
    #     leaf_id = self.get_id()
    #     parent_ids = {leaf_id: parent_id, parent_id: parent_id}
    #     id_pose = {parent_id: edge[0], leaf_id: edge[1]}
    #     all_visited_poses = {edge[0]: parent_id, edge[1]: leaf_id}
    #
    #     # -------------------------------- DFS starts here ---------------
    #     pivot_node = {parent_id: edge[0]}
    #     parent = {leaf_id: parent_id}
    #     visited = [parent_id]
    #     pivot_id = pivot_node.keys()[0]
    #     while not rospy.is_shutdown():
    #         if self.cancel_request:
    #             break
    #         S = [pivot_id]
    #         self.fetch_new_graph()
    #         self.localize_nodes(id_pose, pivot_node)
    #         while len(S) > 0:
    #             u = S.pop()
    #             pivot_node = {parent_ids[u]: id_pose[parent_ids[u]]}
    #             self.move_to_frontier(id_pose[u], theta=pu.theta(id_pose[parent_ids[u]], id_pose[u]))
    #             pu.log_msg(self.robot_id, "Size of stack: {}".format(len(S)), self.debug_mode)
    #             start_time = rospy.Time.now().to_sec()
    #             self.fetch_new_graph()
    #             self.localize_nodes(id_pose, pivot_node)
    #             leaf_pose = id_pose[u]
    #             parent_pose = id_pose[parent_ids[u]]
    #             leaves = self.get_leaves(leaf_pose, parent_pose)
    #             best_leaf = self.get_best_leaf(leaves)
    #             if best_leaf:
    #                 leaf_parent = leaves[best_leaf]
    #                 v_id = self.get_id()
    #                 p_id = self.get_id()
    #                 id_pose[v_id] = best_leaf
    #                 id_pose[p_id] = leaf_parent
    #                 parent_ids[v_id] = p_id
    #                 S.append(v_id)
    #                 parent[v_id] = u
    #             visited.append(u)
    #             all_visited_poses[leaf_pose] = u
    #             end_time = rospy.Time.now().to_sec()
    #             gvg_time = end_time - start_time
    #             self.explore_computation.append({'time': start_time, 'gvg_compute': gvg_time})
    #         pu.log_msg(self.robot_id, "Returned from DFS...", self.debug_mode)
    #         sleep(1)

    def start_gvg_exploration(self, edge):
        parent_id = self.get_id()
        leaf_id = self.get_id()
        parent_ids = {leaf_id: parent_id, parent_id: parent_id}
        id_pose = {parent_id: edge[0], leaf_id: edge[1]}
        all_visited_poses = {edge[0]: parent_id, edge[1]: leaf_id}

        # -------------------------------- DFS starts here ---------------
        pivot_node = {parent_id: edge[0]}
        parent = {leaf_id: parent_id}
        visited = [parent_id]
        pivot_id = pivot_node.keys()[0]
        while not rospy.is_shutdown():
            if self.cancel_request:
                break
            S = [pivot_id]
            self.fetch_new_graph()
            self.localize_nodes(id_pose, pivot_node)
            while len(S) > 0:
                u = S.pop()
                pivot_node = {u: id_pose[u]}
                leaf_pose = id_pose[u]
                self.move_to_frontier(leaf_pose, theta=pu.theta(id_pose[parent_ids[u]], leaf_pose))
                start_time = rospy.Time.now().to_sec()
                self.fetch_new_graph()
                self.localize_nodes(id_pose, pivot_node)
                leaves = self.get_leaves(id_pose[u], id_pose[parent_ids[u]])
                if not leaves:
                    if id_pose[u] not in all_visited_poses:
                        S.append(u)
                    else:
                        pivot_node = {parent_ids[u]: id_pose[parent_ids[u]]}
                else:
                    best_leaf = self.get_best_leaf(leaves)
                    leaf_parent = leaves[best_leaf]
                    v_id = self.get_id()
                    p_id = self.get_id()
                    id_pose[v_id] = best_leaf
                    id_pose[p_id] = leaf_parent
                    parent_ids[v_id] = p_id
                    S.append(v_id)
                    parent[v_id] = u
                    visited.append(u)
                all_visited_poses[leaf_pose] = u
                end_time = rospy.Time.now().to_sec()
                gvg_time = end_time - start_time
                self.explore_computation.append({'time': start_time, 'gvg_compute': gvg_time})
            pu.log_msg(self.robot_id, "Returned from DFS...", self.debug_mode)
            sleep(1)

    def get_id(self):
        id = uuid.uuid4()
        return str(id)

    def line_has_obstacles(self, original_node, new_node):
        points = pu.bresenham_path(original_node, new_node)
        valid_points = [p for p in points if p in self.pixel_desc and self.pixel_desc[p] == OCCUPIED]
        return not len(valid_points)

    def localize_nodes(self, id_pose, pivot_node):
        all_nodes = list(self.adj_list)
        pivot_key = list(pivot_node.keys())[0]
        pivot_pose = pivot_node[pivot_key]
        node_dist = {}
        for n in all_nodes:
            n_dist = pu.D(pivot_pose, n)
            node_dist[n_dist] = n
        new_n = node_dist[min(node_dist)]
        pivot_node[pivot_key] = new_n

        parent = {new_n: None}
        visited = []
        S = deque([new_n])
        node_dist = {}
        while len(S) > 0:
            u = S.popleft()
            if u != pivot_node.values()[0]:
                self.compute_distance(id_pose, node_dist, u)
            neighbors = self.adj_list[u]
            for v in neighbors:
                if v not in visited:
                    S.append(v)
                    parent[v] = u
            visited.append(u)

        node_keys = list(id_pose.keys())
        for k in node_keys:
            min_dist = min(node_dist[k].keys())
            n = node_dist[k][min_dist]
            id_pose[k] = n  # update poses of parent nodes

    def compute_distance(self, id_pose, node_dist, u):
        for k, node in id_pose.items():
            d = pu.D(node, u)
            if self.line_has_obstacles(u, node):
                d = INF
            if k not in node_dist:
                node_dist[k] = {d: u}
            else:
                node_dist[k][d] = u

    def get_leaves(self, node, parent_node):
        leaves = {}
        parent = {node: None}
        visited = [parent_node]
        S = [node]
        while len(S) > 0:
            u = S.pop()
            neighbors = self.adj_list[u]
            for v in neighbors:
                if v not in visited:
                    S.append(v)
                    parent[v] = u
            if len(neighbors) == 1 and parent[u]:
                leaves[u] = parent[u]
            visited.append(u)
        return leaves

    def get_depth(self, leaf, parents):
        l = leaf
        count = 0.0
        while l != None:
            l = parents[l]
            count += 1
        return count

    def get_radius(self, node, parent):
        radius = 1.0
        radii = []
        e = (parent, node)
        e1 = (node, parent)
        if e in self.edges:
            radii.append(pu.D(self.edges[e][0], self.edges[e][1]))
        if e1 in self.edges:
            radii.append(pu.D(self.edges[e1][0], self.edges[e1][1]))
        if radii:
            radius = max(radii)
        return radius

    def is_frontier(self, leaves, leaf_region):
        leaf = list(leaves.keys())[0]
        leaf = pu.get_point(leaf)
        full_cells = {pu.get_point(p): self.pixel_desc[p] for p in leaf_region}
        full_cells[leaf] = FREE
        frontiers = {}
        self.flood_fill(full_cells, None, leaf, [], frontiers)
        return len(frontiers) > 0

    def flood_fill(self, cells, prev_point, new_point, visited, frontiers):
        if new_point not in cells or new_point in visited:
            return
        if prev_point and cells[new_point] != cells[prev_point]:
            if cells[new_point] == UNKNOWN:
                frontiers[prev_point] = new_point
            return
        north_p = list(new_point)
        east_p = list(new_point)
        west_p = list(new_point)
        south_p = list(new_point)
        north_p[INDEX_FOR_X] += 1
        east_p[INDEX_FOR_X] -= 1
        south_p[INDEX_FOR_Y] += 1
        west_p[INDEX_FOR_Y] -= 1
        prev_point = new_point
        visited.append(new_point)

        self.flood_fill(cells, prev_point, tuple(north_p), visited, frontiers)
        self.flood_fill(cells, prev_point, tuple(east_p), visited, frontiers)
        self.flood_fill(cells, prev_point, tuple(south_p), visited, frontiers)
        self.flood_fill(cells, prev_point, tuple(west_p), visited, frontiers)

    def edgelist_callback(self, data):
        self.updated_graph = data

    def fetch_new_graph(self):
        pose = self.get_robot_pose()
        p = Pose()
        p.position.x = pose[INDEX_FOR_X]
        p.position.y = pose[INDEX_FOR_Y]
        response = self.fetch_graph(pose=p)
        self.updated_graph = response.edgelist
        edge = self.process_graph()
        if self.debug_mode:
            marker_pose = (0, 0)
            if self.robot_id == 1:
                marker_pose = (50, 0)
            self.create_markers(marker_pose)
        return edge

    def already_explored(self, v, parent_nodes):
        is_visited = False
        dists = {}
        for node_id, val in parent_nodes.items():
            dists[pu.D(v, val)] = node_id
        closest_distance = min(dists.keys())
        if closest_distance < self.lidar_scan_radius:
            is_visited = True
        return is_visited

    def localize_parent_nodes(self, parent_nodes, all_visited_nodes):
        all_nodes = list(self.adj_list)
        node_keys = list(parent_nodes)
        node_dist = {}
        for n in all_nodes:
            for node_id, val in parent_nodes.items():
                n_dist = pu.D(val, n)
                if node_id not in node_dist:
                    node_dist[node_id] = {n_dist: n}
                else:
                    node_dist[node_id][n_dist] = n
        for k in node_keys:
            n_dists = node_dist[k].keys()
            new_n = node_dist[k][min(n_dists)]
            parent_nodes[k] = new_n
            all_visited_nodes[new_n] = None

    def get_edge(self, ridge):
        edge = {}
        # try:
        p1 = [0.0] * 2
        p1[INDEX_FOR_X] = ridge.nodes[0].position.x
        p1[INDEX_FOR_Y] = ridge.nodes[0].position.y
        p1 = tuple(p1)
        p2 = [0.0] * 2
        p2[INDEX_FOR_X] = ridge.nodes[1].position.x
        p2[INDEX_FOR_Y] = ridge.nodes[1].position.y
        p2 = tuple(p2)
        q1 = [0.0] * 2
        q1[INDEX_FOR_X] = ridge.obstacles[0].position.x
        q1[INDEX_FOR_Y] = ridge.obstacles[0].position.y
        q1 = tuple(q1)
        q2 = [0.0] * 2
        q2[INDEX_FOR_X] = ridge.obstacles[1].position.x
        q2[INDEX_FOR_Y] = ridge.obstacles[1].position.y
        q2 = tuple(q2)
        P = (p1, p2)
        Q = (q1, q2)
        edge[P] = Q
        # except:
        #     rospy.logerr("Invalid goal")

        return edge

    def compute_aux_nodes(self, u, obs):
        ax_nodes = [u]
        q1 = obs[0]
        q2 = obs[1]
        width = pu.D(q1, q2)
        if width >= self.lidar_scan_radius:
            req_poses = width / self.lidar_scan_radius
            ax_nodes += pu.line_points(obs[0], obs[1], req_poses)
        return ax_nodes

    def get_best_leaf(self, leaves):
        polygons, start_end, points, unknown_points, marker_points = self.get_leaf_region(leaves)
        valid_leaves = {}
        for l, region in points.items():
            if self.is_frontier({l: leaves[l]}, region):  # self.line_has_obstacles(v[0], v[1])
                valid_leaves[l] = unknown_points[l]
        if valid_leaves:
            best_leaf = max(valid_leaves, key=valid_leaves.get)
        else:
            best_leaf = max(unknown_points, key=unknown_points.get)
        # -------- #DEBUG --------
        if self.debug_mode:
            marker_pose = (0, 0)
            if self.robot_id == 1:
                marker_pose = (50, 0)
            self.create_marker_array(marker_points, marker_pose)
        # ------- # DEBUG ends here ---------
        return best_leaf

    def area(self, point, orientation, radius):
        unknown_points = []
        radius = int(round(radius))
        for d in np.arange(0, radius, 1):
            distance_points = set()
            for theta in range(-1 * self.lidar_fov // 2, self.lidar_fov + 1):
                angle = np.deg2rad(theta) + orientation
                x = point[INDEX_FOR_X] + d * np.cos(angle)
                y = point[INDEX_FOR_Y] + d * np.sin(angle)
                pt = [0.0] * 2
                pt[INDEX_FOR_X] = x
                pt[INDEX_FOR_Y] = y
                pt = self.round_point(pt)
                distance_points.add(pt)
            for p in distance_points:
                if not pu.is_unknown(p, self.pixel_desc):
                    unknown_points.append(p)
        return unknown_points

    def round_point(self, p):
        xc = round(p[INDEX_FOR_X], 2)
        yc = round(p[INDEX_FOR_Y], 2)
        new_p = [0.0] * 2
        new_p[INDEX_FOR_X] = xc
        new_p[INDEX_FOR_Y] = yc
        new_p = tuple(new_p)
        return new_p

    def create_adjlist(self):
        edge_list = list(self.edges)
        self.adj_list.clear()
        self.leaves.clear()
        edge_dict = {}
        for e in edge_list:
            first = e[0]
            second = e[1]
            edge_dict[(first, second)] = e
            edge_dict[(second, first)] = e
            if first != second:
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
                pair = (list(v)[0], k)
                self.leaves[k] = edge_dict[pair]

    def move_robot_to_goal(self, goal, theta=0):
        self.current_point = goal
        id_val = "robot_{}_{}_explore".format(self.robot_id, self.goal_count)
        self.goal_count += 1
        move = MoveToPosition2DActionGoal()
        frame_id = '/map'.format(self.robot_id)
        move.header.frame_id = frame_id
        move.goal_id.id = id_val
        move.goal.target_pose.x = goal[INDEX_FOR_X]
        move.goal.target_pose.y = goal[INDEX_FOR_Y]
        move.goal.target_pose.theta = theta
        move.goal.header.frame_id = frame_id
        move.goal.target_distance = self.target_distance
        move.goal.target_angle = self.target_angle
        self.moveTo_pub.publish(move)
        self.prev_pose = self.get_robot_pose()
        self.start_time = rospy.Time.now().secs

    def move_to_frontier(self, pose, theta=0):
        scaled_pose = pu.scale_down(pose, self.graph_scale)
        self.moving_to_frontier = True
        self.move_robot_to_goal(scaled_pose, theta)
        while self.moving_to_frontier:
            sleep(1)

    def navigation_plan_callback(self, data):
        if self.waiting_for_plan:
            self.navigation_plan = data.cells

    def move_status_callback(self, data):
        id_0 = "robot_{}_{}_explore".format(self.robot_id, self.goal_count - 1)
        # if data.status_list:
        #     goal_status = data.status_list[0]
        #     if goal_status.goal_id.id:
        #         if goal_status.goal_id.id == id_0:
        #             if goal_status.status == pu.ACTIVE:
        #                 now = rospy.Time.now().secs
        #                 if (now - self.start_time) > 30:  # you can only spend upto 10 sec in same position
        #                     pose = self.get_robot_pose()
        #                     if pu.D(self.prev_pose, pose) < 0.5:
        #                         pu.log_msg(self.robot_id, "Paused for too long", self.debug_mode)
        #                         self.has_arrived = True
        #                         self.moving_to_frontier = False
        #                     self.prev_pose = pose
        #                     self.start_time = now
        #
        #             else:
        #                 if not self.has_arrived:
        #                     self.has_arrived = True
        #                 self.moving_to_frontier = False

    def create_marker_array(self, points, pose):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.position.x = pose[INDEX_FOR_X]
        marker.pose.position.y = pose[INDEX_FOR_Y]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.points = [0.0] * len(points)
        for i in range(len(points)):
            v_pose = pu.scale_down(points[i], self.graph_scale)
            p = Point()
            p.x = v_pose[INDEX_FOR_X]
            p.y = v_pose[INDEX_FOR_Y]
            p.z = 0
            marker.points[i] = p
        self.leaf_publisher.publish(marker)

    def move_result_callback(self, data):
        id_0 = "robot_{}_{}_explore".format(self.robot_id, self.goal_count - 1)
        if data.status:
            if data.status.status == pu.ABORTED:
                pu.log_msg(self.robot_id, "Motion failed", self.debug_mode)
                self.move_to_stop()
                self.motion_failed = True
                self.has_arrived = True
                if self.moving_to_frontier:
                    self.moving_to_frontier = False
                self.current_point = self.get_robot_pose()

            elif data.status.status == pu.SUCCEEDED:
                self.has_arrived = True
                if self.moving_to_frontier:
                    self.moving_to_frontier = False
            if not self.prev_explored:
                self.prev_explored = self.current_point
            self.traveled_distance.append({'time': rospy.Time.now().to_sec(),
                                           'traved_distance': pu.D(self.prev_explored, self.current_point)})
            self.prev_explored = self.current_point

    def move_robot(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.angular.z = vel[1]
        self.pose_publisher.publish(vel_msg)

    def rotate_robot(self):
        current_pose = copy.deepcopy(self.robot_pose)
        self.move_robot((0, 1))
        time.sleep(1)
        while True:
            if self.robot_pose[2] - current_pose[2] < 0.1:
                break
            self.move_robot((0, 1))
        self.move_robot((0, 0))

    def pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = self.get_elevation((orientation.x, orientation.y, orientation.z, orientation.w))
        pose = (position.x, position.y, round(yaw, 2))
        self.robot_pose = pose

    def get_elevation(self, quaternion):
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        return yaw

    def create_markers(self, pose):
        vertices = list(self.leaves)
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.position.x = pose[INDEX_FOR_X]
        marker.pose.position.y = pose[INDEX_FOR_Y]
        marker.pose.position.z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.points = [0.0] * len(vertices)

        for i in range(len(vertices)):
            v_pose = pu.scale_down(vertices[i], self.graph_scale)
            p = Point()
            p.x = v_pose[INDEX_FOR_X]
            p.y = v_pose[INDEX_FOR_Y]
            p.z = 0
            marker.points[i] = p
        self.vertex_publisher.publish(marker)

        # # Publish the edges
        edges = list(self.edges)
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.03

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.points = [0.0] * len(edges) * 2

        for i in range(len(edges)):
            edge = edges[i]
            v1 = pu.scale_down(edge[0], self.graph_scale)
            v2 = pu.scale_down(edge[1], self.graph_scale)
            p1 = Point()
            p1.x = v1[INDEX_FOR_X]
            p1.y = v1[INDEX_FOR_Y]
            p1.z = 0
            marker.points[2 * i] = p1

            p2 = Point()
            p2.x = v2[INDEX_FOR_X]
            p2.y = v2[INDEX_FOR_Y]
            p2.z = 0
            marker.points[2 * i + 1] = p2

        self.edge_publisher.publish(marker)

    def get_leaf_region(self, leaves):
        polygons = {}
        start_end = {}
        for leaf, parent in leaves.items():
            width = self.get_radius(leaf, parent)
            radius = self.lidar_scan_radius

            x = leaf[0]
            y = leaf[1]

            opp = width / 2.0
            adj = radius
            hyp = np.sqrt(opp ** 2 + adj ** 2)
            theta1 = pu.theta(parent, leaf)
            angle_sum = (np.pi / 2) + theta1
            cos_val = opp * np.cos(angle_sum)
            sin_val = opp * np.sin(angle_sum)

            top_left_x = x + cos_val
            top_left_y = y + sin_val

            bottom_left_x = x - cos_val
            bottom_left_y = y - sin_val

            lx = x + hyp * np.cos(theta1)
            ly = y + hyp * np.sin(theta1)

            top_right_x = lx + cos_val
            top_right_y = ly + sin_val

            bottom_right_x = lx - cos_val
            bottom_right_y = ly - sin_val

            polygon = Polygon([(bottom_left_x, bottom_left_y), (top_left_x, top_left_y), (top_right_x, top_right_y),
                               (bottom_right_x, bottom_right_y)])

            polygons[leaf] = polygon
            start_end[leaf] = (leaf, (lx, ly))

        points = {}
        unknown_points = {}
        leaf_keys = leaves.keys()
        marker_points = []
        for pose, value in self.pixel_desc.items():
            point = sg.Point(pose[INDEX_FOR_X], pose[INDEX_FOR_Y])
            for leaf in leaf_keys:
                if polygons[leaf].contains(point):
                    marker_points.append(pose)
                    if leaf not in points:
                        points[leaf] = [pose]
                    else:
                        points[leaf].append(pose)
                    if value == UNKNOWN:
                        if leaf not in unknown_points:
                            unknown_points[leaf] = 1
                        else:
                            unknown_points[leaf] += 1
        return polygons, start_end, points, unknown_points, marker_points

    def get_robot_pose(self):
        robot_pose = None
        while not robot_pose:
            try:
                self.listener.waitForTransform("/map".format(self.robot_id),
                                               "/base_link".format(self.robot_id), rospy.Time(),
                                               rospy.Duration(4.0))
                (robot_loc_val, rot) = self.listener.lookupTransform("/map".format(self.robot_id),
                                                                     "/base_link".format(self.robot_id),
                                                                     rospy.Time(0))
                robot_pose = (math.floor(robot_loc_val[0]), math.floor(robot_loc_val[1]), robot_loc_val[2])
                time.sleep(1)
            except:
                pass
        return robot_pose

    def save_all_data(self):
        pu.save_data(self.traveled_distance,
                     '{}/traveled_distance_{}_{}_{}_{}_{}.pickle'.format(self.method, self.environment,
                                                                         self.robot_count,
                                                                         self.run,
                                                                         self.termination_metric,
                                                                         self.robot_id))
        pu.save_data(self.explore_computation,
                     '{}/explore_computation_{}_{}_{}_{}_{}.pickle'.format(self.method, self.environment,
                                                                           self.robot_count,
                                                                           self.run,
                                                                           self.termination_metric,
                                                                           self.robot_id))


if __name__ == "__main__":
    graph = GVGExplore()
    rospy.spin()
