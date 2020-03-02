#!/usr/bin/python
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import pickle
from os import path
import heapq
from project_utils import D, get_ridge_desc, line_points, pixel2pose, pose2pixel, theta, get_point, process_edges, W, \
    get_vector, compute_similarity, there_is_unknown_region, collinear, scale_up, scale_down, SCALE, INDEX_FOR_X, \
    INDEX_FOR_Y, ACTIVE, SUCCEEDED, ABORTED, save_data
import actionlib
import rospy
import math
from gvgexploration.msg import GvgExploreFeedback, GvgExploreGoal, GvgExploreResult, EdgeList, Coverage, \
    GvgExploreAction
from gvgexploration.srv import *
from nav2d_navigator.msg import MoveToPosition2DActionGoal, MoveToPosition2DActionResult
from actionlib_msgs.msg import GoalStatusArray, GoalID
from std_srvs.srv import Trigger
from nav_msgs.msg import GridCells, Odometry
from geometry_msgs.msg import Twist, Pose
import time
import tf
from std_msgs.msg import String

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
        self.is_exploring = False
        self.moving_to_frontier = False
        self.is_processing_graph = False
        self.start_time = 0
        self.prev_pose = 0
        rospy.init_node("gvg_explore", anonymous=True)
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
        self.opposite_vector_bias = rospy.get_param("~opposite_vector_bias") * SCALE
        self.target_distance = rospy.get_param('~target_distance')
        self.target_angle = rospy.get_param('~target_angle')
        self.robot_count = rospy.get_param("~robot_count")
        self.environment = rospy.get_param("~environment")
        self.max_coverage_ratio = rospy.get_param("~max_coverage")
        rospy.Subscriber("/robot_{}/MoveTo/status".format(self.robot_id), GoalStatusArray, self.move_status_callback)
        rospy.Subscriber("/robot_{}/MoveTo/result".format(self.robot_id), MoveToPosition2DActionResult,
                         self.move_result_callback)
        rospy.Subscriber("/robot_{}/navigator/plan".format(self.robot_id), GridCells, self.navigation_plan_callback)
        rospy.Subscriber("/robot_{}/edge_list".format(self.robot_id), EdgeList, self.process_edges)
        self.move_to_stop = rospy.ServiceProxy('/robot_{}/Stop'.format(self.robot_id), Trigger)
        self.moveTo_pub = rospy.Publisher("/robot_{}/MoveTo/goal".format(self.robot_id), MoveToPosition2DActionGoal,
                                          queue_size=10)
        self.fetch_graph = rospy.ServiceProxy('/robot_{}/fetch_graph'.format(self.robot_id), FetchGraph)
        self.pose_publisher = rospy.Publisher("/robot_{}/cmd_vel".format(self.robot_id), Twist, queue_size=1)
        rospy.Subscriber("/robot_{}/odom".format(self.robot_id), Odometry, callback=self.pose_callback)
        self.exploration_feedback = GvgExploreFeedback()
        self.exploration_result = GvgExploreResult()
        self.action_server = actionlib.SimpleActionServer('/robot_{}/gvg_explore'.format(self.robot_id),
                                                          GvgExploreAction, execute_cb=self.start_gvg_exploration,
                                                          auto_start=False)
        self.action_server.start()
        self.listener = tf.TransformListener()
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)
        self.already_shutdown = False
        rospy.loginfo("Robot {}: Exploration server online...".format(self.robot_id))

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            r.sleep()

    def process_edges(self, data):
        edgelist = data.edgelist.edges
        pixels = data.edgelist.pixels
        self.edges.clear()
        self.pixel_desc.clear()
        for r in edgelist:
            edge = self.get_edge(r)
            self.edges.update(edge)
        for p in pixels:
            point = [0.0] * 2
            point[INDEX_FOR_X] = p.x
            point[INDEX_FOR_Y] = p.y
            point = tuple(point)
            self.pixel_desc[point] = p.desc
        self.create_adjlist()

    def start_gvg_exploration(self, goal):
        success = True
        visited_nodes = {}
        robot_pose = (goal.pose.position.x, goal.pose.position.y, goal.pose.position.z)
        self.move_to_frontier(robot_pose, visited_nodes)
        while self.coverage_ratio < 0.8:
            self.run_dfs(visited_nodes)
        if success:
            self.create_result(visited_nodes)
            self.action_server.set_succeeded(self.exploration_result)

    def move_to_frontier(self, goal, visited):
        self.moving_to_frontier = True
        self.move_robot_to_goal(goal)
        # rospy.logerr("Robot {} going to frontier....".format(self.robot_id))
        while self.moving_to_frontier:
            continue
        self.create_feedback(self.get_robot_pose())
        visited[scale_up(goal)] = None

    def run_dfs(self, visited_nodes):
        robot_pose = self.get_robot_pose()
        p = Pose()
        p.position.x = robot_pose[INDEX_FOR_X]
        p.position.y = robot_pose[INDEX_FOR_Y]
        edge_data = self.fetch_graph(FetchGraphRequest(pose=p))
        self.process_edges(edge_data)
        pose = scale_up(robot_pose)
        edge = self.get_closest_ridge(pose, visited_nodes)
        if edge:
            leaf = edge[1]
            leaf_parent = edge[0]
            if D(pose, leaf_parent) > D(pose, leaf):
                leaf = edge[0]
                leaf_parent = edge[1]
            S = [leaf]
            parents = {leaf: leaf_parent}
            # --------- DFS -------
            # rospy.logerr("Robot {}: Starting DFS..".format(self.robot_id))
            while len(S) > 0:
                u = S.pop()
                if self.debug_mode:
                    if self.robot_id == 0:
                        pose = self.get_robot_pose()
                        point = scale_up(pose)
                        self.plot_intersections(point, u, parents[u])
                self.explore_point(u)
                visited_nodes[u] = None
                child_leaf = self.get_child_leaf(u, parents[u], visited_nodes)
                if child_leaf:
                    S.append(child_leaf[1])
                    parents[child_leaf[1]] = child_leaf[0]

    def get_closest_ridge(self, robot_pose, all_visited):
        closest_ridge = {}
        edge_list = list(self.edges)
        close_edge = None
        for e in edge_list:
            p1 = e[0]
            p2 = e[1]
            if D(robot_pose, p1) > D(robot_pose, p2):
                p1 = e[1]
                p2 = e[0]
            if not self.is_visited(p1, all_visited) and not self.is_visited(p2, all_visited):
                d = max([D(robot_pose, e[0]), D(robot_pose, e[1])])
                closest_ridge[e] = d
            else:
                all_visited[p1] = None
                all_visited[p2] = None
        if closest_ridge:
            close_edge = max(closest_ridge, key=closest_ridge.get)
        return close_edge

    def get_closest_leaf(self, pose, visited_nodes):
        close_edge = None
        closest_leaf_edge = {}
        edge_list = list(self.leaves)
        for e in edge_list:
            p = e[1]
            if not self.is_visited(p, visited_nodes):
                closest_leaf_edge[e] = D(pose, p)
        if closest_leaf_edge:
            close_edge = max(closest_leaf_edge, key=closest_leaf_edge.get)

        return close_edge

    def get_edge(self, goal):
        edge = {}
        try:
            p1 = [0.0] * 2
            p1[INDEX_FOR_X] = goal.px[0]
            p1[INDEX_FOR_Y] = goal.py[0]
            p1 = tuple(p1)
            p2 = [0.0] * 2
            p2[INDEX_FOR_X] = goal.px[1]
            p2[INDEX_FOR_Y] = goal.py[1]
            p2 = tuple(p2)
            q1 = [0.0] * 2
            q1[INDEX_FOR_X] = goal.px[2]
            q1[INDEX_FOR_Y] = goal.py[2]
            q1 = tuple(q1)
            q2 = [0.0] * 2
            q2[INDEX_FOR_X] = goal.px[3]
            q2[INDEX_FOR_Y] = goal.py[3]
            q2 = tuple(q2)
            P = (p1, p2)
            Q = (q1, q2)
            edge[P] = Q
        except:
            rospy.logerr("Invalid goal")

        return edge

    def create_feedback(self, point):
        next_point = scale_down(point)
        self.exploration_feedback.progress = "Starting exploration"
        pose = Pose()
        pose.position.x = next_point[INDEX_FOR_X]
        pose.position.y = next_point[INDEX_FOR_Y]
        self.exploration_feedback.current_point = pose
        self.action_server.publish_feedback(self.exploration_feedback)

    def create_result(self, visited_nodes):
        all_nodes = list(visited_nodes)
        allposes = []
        for n in all_nodes:
            point = scale_down(n)
            pose = Pose()
            pose.position.x = point[INDEX_FOR_X]
            pose.position.y = point[INDEX_FOR_Y]
            allposes.append(pose)
        self.exploration_result.result = "gvg exploration complete"
        self.exploration_result.explored_points = allposes

    def is_visited(self, v, visited):
        already_visited = False
        if v in visited:
            return True
        return already_visited

    def compute_aux_nodes(self, u, obs):
        ax_nodes = [u]
        q1 = obs[0]
        q2 = obs[1]
        width = D(q1, q2)
        if width >= self.lidar_scan_radius:
            req_poses = width / self.lidar_scan_radius
            ax_nodes += line_points(obs[0], obs[1], req_poses)
        return ax_nodes

    def get_child_leaf(self, first_node, parent_node, all_visited):
        parent = {first_node: parent_node}
        local_visited = [parent_node]
        local_leaves = []
        S = [first_node]
        next_leaf = None
        while len(S) > 0:
            u = S.pop()
            if u in self.adj_list:
                neighbors = self.adj_list[u]
                no_neighbor = True
                for v in neighbors:
                    if not self.is_visited(v, all_visited):
                        if not self.is_visited(v, local_visited):
                            no_neighbor = False
                            S.append(v)
                            parent[v] = u
                if no_neighbor and len(neighbors) == 1:
                    nei = neighbors.pop()
                    local_leaves.append((nei, u))
                local_visited.append(u)
        dists = {}
        for l in local_leaves:
            if self.is_near_unexplored_aread(l[1]):
                dists[D(first_node, l[1])] = l
        if dists:
            next_leaf = dists[min(dists.keys())]
        return next_leaf

    def is_near_unexplored_aread(self, point):
        unknwon_neighborhood = 0
        x = point[INDEX_FOR_X]
        y = point[INDEX_FOR_Y]
        for r in range(self.lidar_scan_radius // 2):
            for angle in range(self.lidar_fov):
                new_p = [0.0] * 2
                new_p[INDEX_FOR_X] = round(x + r * np.cos(np.radians(angle)))
                new_p[INDEX_FOR_Y] = round(y + r * np.sin(np.radians(angle)))
                new_p = tuple(new_p)
                if new_p in self.pixel_desc and self.pixel_desc[new_p] == UNKNOWN:
                    unknwon_neighborhood += 1
        return unknwon_neighborhood >= self.lidar_scan_radius

    def explore_point(self, goal_pose):
        explored_pose = scale_down(goal_pose)
        self.has_arrived = False
        self.move_attempt = 0
        self.move_robot_to_goal(explored_pose)
        while not self.has_arrived and not self.navigation_failed:
            rospy.sleep(1)
        self.has_arrived = False
        self.navigation_failed = False

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
                self.leaves[pair] = edge_dict[pair]

    def move_robot_to_goal(self, goal, direction=1):
        self.current_point = goal
        id_val = "robot_{}_{}_explore".format(self.robot_id, self.goal_count)
        move = MoveToPosition2DActionGoal()
        frame_id = '/robot_{}/map'.format(self.robot_id)
        move.header.frame_id = frame_id
        move.goal_id.id = id_val
        move.goal.target_pose.x = goal[INDEX_FOR_X]
        move.goal.target_pose.y = goal[INDEX_FOR_Y]
        move.goal.header.frame_id = frame_id
        move.goal.target_distance = self.target_distance
        move.goal.target_angle = self.target_angle
        self.moveTo_pub.publish(move)
        self.prev_pose = self.get_robot_pose()
        self.start_time = rospy.Time.now().secs
        self.goal_count += 1

    def navigation_plan_callback(self, data):
        if self.waiting_for_plan:
            self.navigation_plan = data.cells

    def move_status_callback(self, data):
        id_0 = "robot_{}_{}_explore".format(self.robot_id, self.goal_count - 1)
        if data.status_list:
            goal_status = data.status_list[0]
            if goal_status.goal_id.id:
                if goal_status.goal_id.id == id_0:
                    if goal_status.status == ACTIVE:
                        now = rospy.Time.now().secs
                        if (now - self.start_time) > 5:
                            pose = self.get_robot_pose()
                            if D(self.prev_pose, pose) < 0.5:
                                self.has_arrived = True
                            self.prev_pose = pose
                            self.start_time = now
                    else:
                        if not self.has_arrived:
                            self.has_arrived = True

    def move_result_callback(self, data):
        id_0 = "robot_{}_{}_explore".format(self.robot_id, self.goal_count - 1)
        if data.status:
            if data.status.status == ABORTED:
                # if data.status.goal_id.id == id_0:
                if self.move_attempt < MAX_ATTEMPTS:
                    self.rotate_robot()
                    self.move_robot_to_goal(self.current_point, TO_FRONTIER)
                    self.move_attempt += 1
                else:
                    # rospy.logerr("Robot {} can't reach goal: {}".format(self.robot_id, self.current_point))
                    self.navigation_failed = True
                    if self.moving_to_frontier:
                        self.moving_to_frontier = False
            elif data.status.status == SUCCEEDED:
                if data.status.goal_id.id == id_0:
                    # rospy.logerr("Robot {} arrived at goal..".format(self.robot_id))
                    # self.rotate_robot()
                    self.has_arrived = True
                    if self.moving_to_frontier:
                        self.moving_to_frontier = False
                    if not self.prev_explored:
                        self.prev_explored = self.current_point
                    self.traveled_distance.append({'time': rospy.Time.now().to_sec(),
                                                   'traved_distance': D(self.prev_explored, self.current_point)})
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

    def plot_intersections(self, robot_pose, next_leaf, parent_leaf):
        fig, ax = plt.subplots(figsize=(16, 10))
        x_pairs, y_pairs = process_edges(self.edges)
        for i in range(len(x_pairs)):
            x = x_pairs[i]
            y = y_pairs[i]
            ax.plot(x, y, "g-.")

        leaves = [v[1] for v in list(self.leaves)]
        lx, ly = zip(*leaves)
        ax.scatter(lx, ly, marker='*', color='purple')
        ax.scatter(robot_pose[INDEX_FOR_X], robot_pose[INDEX_FOR_Y], marker='*', color='blue')
        ax.scatter(next_leaf[INDEX_FOR_X], next_leaf[INDEX_FOR_Y], marker='*', color='green')
        ax.plot([next_leaf[INDEX_FOR_X], parent_leaf[INDEX_FOR_X]], [next_leaf[INDEX_FOR_Y], parent_leaf[INDEX_FOR_Y]],
                'r-.')
        plt.grid()
        plt.savefig("gvg/current_leaves_{}_{}_{}.png".format(self.robot_id, time.time(), self.run))
        plt.close(fig)
        # plt.show()

    def shutdown_callback(self, msg):
        if not self.already_shutdown:
            rospy.signal_shutdown('GVG Explore: Shutdown command received!')
            pass

    def save_all_data(self):
        save_data(self.traveled_distance,
                  'gvg/traveled_distance_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                                       self.termination_metric, self.robot_id))


if __name__ == "__main__":
    graph = GVGExplore()
    rospy.spin()
