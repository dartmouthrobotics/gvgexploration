#!/usr/bin/python
from threading import Thread

import matplotlib
import uuid
import shapely.geometry as sg
import shapely.affinity as sa

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
        self.start_time = 0
        self.prev_pose = 0
        self.updated_graph = None
        rospy.init_node("gvgexplore", anonymous=True)
        self.robot_id = rospy.get_param('~robot_id')
        self.run = rospy.get_param("~run")
        self.debug_mode = rospy.get_param("~debug_mode")
        self.termination_metric = rospy.get_param("~termination_metric")
        self.min_edge_length = rospy.get_param("~min_edge_length".format(self.robot_id)) * SCALE
        self.min_hallway_width = rospy.get_param("~min_hallway_width".format(self.robot_id)) * SCALE
        self.comm_range = rospy.get_param("~comm_range".format(self.robot_id)) * SCALE
        self.point_precision = rospy.get_param("~point_precision".format(self.robot_id))
        self.lidar_scan_radius = rospy.get_param("~lidar_scan_radius".format(self.robot_id)) * SCALE
        self.lidar_fov = rospy.get_param("~lidar_fov")
        self.slope_bias = rospy.get_param("~slope_bias") * SCALE
        self.separation_bias = rospy.get_param("~separation_bias".format(self.robot_id)) * SCALE
        self.opposite_vector_bias = rospy.get_param("~opposite_vector_bias")  # * SCALE
        self.target_distance = rospy.get_param('~target_distance')
        self.target_angle = rospy.get_param('~target_angle')
        self.robot_count = rospy.get_param("~robot_count")
        self.environment = rospy.get_param("~environment")
        self.max_coverage_ratio = rospy.get_param("~max_coverage")
        rospy.Subscriber("/robot_{}/MoveTo/status".format(self.robot_id), GoalStatusArray, self.move_status_callback)
        rospy.Subscriber("/robot_{}/MoveTo/result".format(self.robot_id), MoveToPosition2DActionResult,
                         self.move_result_callback)
        rospy.Subscriber("/robot_{}/navigator/plan".format(self.robot_id), GridCells, self.navigation_plan_callback)
        rospy.Subscriber("/robot_{}/edge_list".format(self.robot_id), EdgeList, self.edgelist_callback)
        self.move_to_stop = rospy.ServiceProxy('/robot_{}/Stop'.format(self.robot_id), Trigger)
        self.moveTo_pub = rospy.Publisher("/robot_{}/MoveTo/goal".format(self.robot_id), MoveToPosition2DActionGoal,
                                          queue_size=10)
        self.vertex_publisher = rospy.Publisher("/robot_{}/explore/vertices".format(self.robot_id), Marker,
                                                queue_size=10)
        self.edge_publisher = rospy.Publisher("/robot_{}/explore/edges".format(self.robot_id), Marker, queue_size=10)
        self.fetch_graph = rospy.ServiceProxy('/robot_{}/fetch_graph'.format(self.robot_id), FetchGraph)
        self.pose_publisher = rospy.Publisher("/robot_{}/cmd_vel".format(self.robot_id), Twist, queue_size=1)
        rospy.Subscriber("/robot_{}/odom".format(self.robot_id), Odometry, callback=self.pose_callback)
        rospy.Subscriber('/robot_{}/gvgexplore/goal'.format(self.robot_id), Ridge, self.initial_action_handler)
        rospy.Service('/robot_{}/gvgexplore/cancel'.format(self.robot_id), CancelExploration,
                      self.received_prempt_handler)
        self.goal_feedback_pub = rospy.Publisher("/robot_{}/gvgexplore/feedback".format(self.robot_id), Pose,
                                                 queue_size=1)

        self.listener = tf.TransformListener()
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)
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
        scaled_pose = pu.scale_down(leaf)
        self.moving_to_frontier = True
        self.move_robot_to_goal(scaled_pose, 0)
        while self.moving_to_frontier:
            sleep(1)
        p = self.get_robot_pose()
        pose = Pose()
        pose.position.x = p[INDEX_FOR_X]
        pose.position.y = p[INDEX_FOR_Y]
        self.goal_feedback_pub.publish(pose)  # publish to feedback
        self.start_gvg_exploration(edge)

    def start_gvg_exploration(self, edge):
        current_edge = edge
        while not rospy.is_shutdown():
            current_edge = self.run_other_dfs(current_edge)
            self.fetch_new_graph()
            if self.cancel_request:
                break
            current_edge = (current_edge[1], current_edge[0])
            sleep(1)
        self.cancel_request = False

    def move_to_frontier(self, pose, theta=0):
        scaled_pose = pu.scale_down(pose)
        self.moving_to_frontier = True
        self.move_robot_to_goal(scaled_pose, theta)
        while self.moving_to_frontier:
            sleep(1)

    def run_other_dfs(self, edge):
        actual_visited_nodes = {}
        visited_nodes = {}
        node_id = self.get_id()
        parent_nodes = {node_id: edge[0]}
        stack_nodes = {node_id: edge[1]}
        parent = {node_id: None}
        visited = []
        S = [node_id]
        last_edge = edge
        while len(S) > 0:
            u = S.pop()
            if self.cancel_request:
                break
            last_edge = (stack_nodes[u], parent_nodes[u])
            leave_node = stack_nodes[u]
            ancestor_id = parent[u]
            angle = 0
            if parent[u]:
                angle = pu.theta(parent_nodes[ancestor_id], leave_node)
            self.move_to_frontier(leave_node, theta=angle)
            start_time = rospy.Time.now().to_sec()
            visited_nodes[leave_node] = None
            self.fetch_new_graph()
            self.localize_parent_nodes(parent_nodes, actual_visited_nodes)
            ancestor_node = None
            if parent[u]:
                ancestor_node = parent_nodes[parent[u]]
            leave_node = self.localize_leaf_node(ancestor_node, parent_nodes[u], stack_nodes[u])
            end_time = rospy.Time.now().to_sec()
            gvg_time = end_time - start_time
            self.explore_computation.append({'time': start_time, 'gvg_compute': gvg_time})
            if leave_node:
                next_leaves = self.get_leaves(leave_node, parent_nodes[u])
                for leaf, lp in next_leaves.items():
                    if leaf not in visited_nodes and leaf not in actual_visited_nodes and self.is_near_unexplored_area(
                            leaf, lp):
                        v_id = self.get_id()
                        S.append(v_id)
                        parent[v_id] = u
                        stack_nodes[v_id] = leaf
                        parent_nodes[v_id] = lp
            visited.append(u)
        return last_edge

    def get_leaves(self, node, parent_node):
        leaves = {}
        parent = {node: parent_node}
        visited = [parent_node]
        S = [node]
        while len(S) > 0:
            u = S.pop()
            neighbors = self.adj_list[u]
            for v in neighbors:
                if v not in visited:
                    S.append(v)
                    parent[v] = u
                    child_nodes = self.adj_list[v]
                    if len(child_nodes) <= 1:
                        leaves[v] = u
            visited.append(u)
        return leaves

    def edgelist_callback(self, data):
        self.updated_graph = data

    def fetch_new_graph(self):
        edge = self.process_graph()
        if self.debug_mode:
            marker_pose = (0, 0)
            if self.robot_id == 1:
                marker_pose = (100, 0)
            self.create_markers(marker_pose)
        return edge

    def already_explored(self, v, parent_nodes):
        is_visited = False
        dists = {}
        for node_id, val in parent_nodes.items():
            dists[pu.D(v, val)] = node_id
        closest_distance = min(dists.keys())
        if closest_distance < 2.0:
            is_visited = True
        return is_visited

    def localize_parent_nodes(self, parent_nodes, actual_visited_nodes):
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
            actual_visited_nodes[new_n] = None

    def localize_leaf_node(self, ancestor_node, parent_node, leaf_node):
        node_dists = {}
        visited = [parent_node]
        if ancestor_node:
            visited = [ancestor_node]
            parent = {ancestor_node: None}
        else:
            parent = {parent_node: None}
        S = [parent_node]
        while len(S) > 0:
            u = S.pop()
            neighbors = self.adj_list[u]
            for v in neighbors:
                if v not in visited:
                    S.append(v)
                    parent[v] = u
                    node_dists[pu.D(leaf_node, v)] = u
            visited.append(u)
        node = None
        if node_dists:
            node = node_dists[min(node_dists.keys())]
        return node

    def get_id(self):
        id = uuid.uuid4()
        return str(id)

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

    def create_ridge(self, ridge_dict):
        edge = ridge_dict.keys()[0]
        obs = ridge_dict.values()[0]
        p1 = Pose()
        p1.position.x = edge[0][INDEX_FOR_X]
        p1.position.y = edge[0][INDEX_FOR_Y]

        p2 = Pose()
        p2.position.x = edge[1][INDEX_FOR_X]
        p2.position.y = edge[1][INDEX_FOR_Y]

        q1 = Pose()
        q1.position.x = obs[0][INDEX_FOR_X]
        q1.position.y = obs[0][INDEX_FOR_Y]

        q2 = Pose()
        q2.position.x = obs[1][INDEX_FOR_X]
        q2.position.y = obs[1][INDEX_FOR_Y]

        r = Ridge()
        r.nodes = [p1, p2]
        r.obstacles = [q1, q2]
        return r

    def is_visited(self, v, visited):
        already_visited = False
        if v in visited:
            already_visited = True
        return already_visited

    def compute_aux_nodes(self, u, obs):
        ax_nodes = [u]
        q1 = obs[0]
        q2 = obs[1]
        width = pu.D(q1, q2)
        if width >= self.lidar_scan_radius:
            req_poses = width / self.lidar_scan_radius
            ax_nodes += pu.line_points(obs[0], obs[1], req_poses)
        return ax_nodes

    def get_child_leaf(self, first_node, parent_node, all_visited):
        parents = {first_node: None}
        local_visited = [parent_node]
        leaves = []
        S = [first_node]
        next_leaf = None
        while len(S) > 0:
            u = S.pop()
            neighbors = self.adj_list[u]
            for v in neighbors:
                if v not in local_visited and v != u:
                    S.append(v)
                    parents[v] = u
                    if v in self.leaves and v not in all_visited and self.is_near_unexplored_area(v, u):
                        leaves.append(v)
            local_visited.append(u)

        dists = {}
        for l in leaves:
            d = pu.D(first_node, l)  # self.get_path(l, parents)
            dists[d] = l
        if dists:
            next_leaf = dists[min(dists.keys())]

        return next_leaf

    def get_path(self, l, parents):
        n = l
        count = 0
        while parents[n] is not None:
            n = parents[n]
            # rospy.logerr("Count: {}, Node: {}".format(count, n))
            count += 1
        return count

    def is_near_unexplored_area(self, node, parent_node):
        start = time.time()
        unknown_neighborhood = 0
        point = sg.Point(node[INDEX_FOR_X], node[INDEX_FOR_Y])
        circle = point.buffer(self.lidar_scan_radius)
        region_obstacles = []
        for k, v in self.pixel_desc.items():
            geo_p = sg.Point(k[INDEX_FOR_X], k[INDEX_FOR_Y])
            if circle.contains(geo_p):
                if v == OCCUPIED:
                    region_obstacles.append(k)
                else:
                    unknown_neighborhood += 1
        # return unknown_neighborhood > self.lidar_scan_radius
        unknown_area_close = True
        obs_dist = {}
        if unknown_neighborhood:
            v1 = pu.get_vector(parent_node, node)
            for ob in region_obstacles:
                v2 = pu.get_vector(node, ob)
                distance = pu.D(node, ob)
                cos_theta, separation = pu.compute_similarity(v1, v2, (parent_node, node), (node, ob))
                if 1 - self.opposite_vector_bias <= cos_theta <= 1:
                    unknown_area_close = False
                    break
        else:
            unknown_area_close = False
        diff = time.time() - start
        # rospy.logerr("Time consumed: {}".format(diff))
        return unknown_area_close

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
                pair = (v.pop(), k)
                self.leaves[k] = edge_dict[pair]

    def move_robot_to_goal(self, goal, theta=0):
        self.current_point = goal
        id_val = "robot_{}_{}_explore".format(self.robot_id, self.goal_count)
        self.goal_count += 1
        move = MoveToPosition2DActionGoal()
        frame_id = '/robot_{}/map'.format(self.robot_id)
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

    def navigation_plan_callback(self, data):
        if self.waiting_for_plan:
            self.navigation_plan = data.cells

    def move_status_callback(self, data):
        id_0 = "robot_{}_{}_explore".format(self.robot_id, self.goal_count - 1)
        if data.status_list:
            goal_status = data.status_list[0]
            # if goal_status.goal_id.id:
            #     if goal_status.goal_id.id == id_0:
            #         if goal_status.status == pu.ACTIVE:
            #             now = rospy.Time.now().secs
            #             if (now - self.start_time) > 30:  # you can only spend upto 10 sec in same position
            #                 pose = self.get_robot_pose()
            #                 if pu.D(self.prev_pose, pose) < 0.5:
            #                     pu.log_msg(self.robot_id, "Paused for too long", self.debug_mode)
            #                     self.has_arrived = True
            #                     self.moving_to_frontier = False
            #                 self.prev_pose = pose
            #                 self.start_time = now
            #
            #         else:
            #             if not self.has_arrived:
            #                 self.has_arrived = True
            #             self.moving_to_frontier = False

    def move_result_callback(self, data):
        id_0 = "robot_{}_{}_explore".format(self.robot_id, self.goal_count - 1)
        if data.status:
            if data.status.status == pu.ABORTED:
                pu.log_msg(self.robot_id, "Motion failed", self.debug_mode)
                # if data.status.goal_id.id == id_0:
                self.has_arrived = True
                if self.moving_to_frontier:
                    self.moving_to_frontier = False
                self.current_point = self.get_robot_pose()

            elif data.status.status == pu.SUCCEEDED:
                # if data.status.goal_id.id == id_0:
                self.has_arrived = True
                if self.moving_to_frontier:
                    self.moving_to_frontier = False
            if not self.prev_explored:
                self.prev_explored = self.current_point
            self.traveled_distance.append({'time': rospy.Time.now().to_sec(),
                                           'traved_distance': pu.D(self.prev_explored, self.current_point)})
            self.prev_explored = self.current_point

    # def chosen_point_callback(self, data):
    #     self.received_choices[(data.x, data.y)] = data

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
            v_pose = pu.scale_down(vertices[i])
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
            v1 = pu.scale_down(edge[0])
            v2 = pu.scale_down(edge[1])
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
                time.sleep(1)
            except:
                # rospy.logerr("Robot {}: Can't fetch robot pose from tf".format(self.robot_id))
                pass

        return robot_pose

    def plot_intersections(self, robot_pose, next_leaf, close_edge):
        fig, ax = plt.subplots(figsize=(16, 10))
        x_pairs, y_pairs = pu.process_edges(self.edges)

        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "g-.")
        leaves = [v for v in list(self.leaves)]
        lx, ly = zip(*leaves)
        ax.scatter(lx, ly, marker='*', color='purple')
        ax.scatter(robot_pose[INDEX_FOR_X], robot_pose[INDEX_FOR_Y], marker='*', color='blue')
        ax.scatter(next_leaf[INDEX_FOR_X], next_leaf[INDEX_FOR_Y], marker='*', color='green')
        ax.plot([close_edge[0][INDEX_FOR_X], close_edge[1][INDEX_FOR_X]],
                [close_edge[0][INDEX_FOR_Y], close_edge[1][INDEX_FOR_Y]], 'r-.')
        plt.grid()
        plt.savefig("gvgexplore/current_leaves_{}_{}_{}.png".format(self.robot_id, time.time(), self.run))
        plt.close(fig)
        # plt.show()

    def shutdown_callback(self, msg):
        if not self.already_shutdown:
            self.save_all_data()
            rospy.signal_shutdown('GVG Explore: Shutdown command received!')
            pass

    def save_all_data(self):
        pu.save_data(self.traveled_distance,
                     'gvg/traveled_distance_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count,
                                                                          self.run,
                                                                          self.termination_metric,
                                                                          self.robot_id))
        pu.save_data(self.explore_computation,
                     'gvg/explore_computation_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count,
                                                                            self.run,
                                                                            self.termination_metric,
                                                                            self.robot_id))


if __name__ == "__main__":
    graph = GVGExplore()
    rospy.spin()
