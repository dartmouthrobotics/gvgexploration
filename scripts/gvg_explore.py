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
    INDEX_FOR_Y
import actionlib
import rospy
import math
from gvgexploration.msg import GvgExploreFeedback, GvgExploreGoal, GvgExploreResult, EdgeList, \
    ChosenPoint, BufferedData
from gvgexploration.msg import *
from nav2d_navigator.msg import MoveToPosition2DActionGoal, MoveToPosition2DActionResult
from actionlib_msgs.msg import GoalStatusArray, GoalID
from std_srvs.srv import Trigger
from nav_msgs.msg import GridCells, Odometry
from geometry_msgs.msg import Twist, Pose
import time
import tf

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
ACTIVE_STATE = 1  # This state shows that the robot is collecting messages
PASSIVE_STATE = -1  # This state shows  that the robot is NOT collecting messages
ACTIVE = 1  # The goal is currently being processed by the action server
SUCCEEDED = 3  # The goal was achieved successfully by the action server (Terminal State)
ABORTED = 4  # The goal was aborted during execution by the action server due to some failure (Terminal State)
LOST = 9  # An action client can determine that a goal is LOST. This should not be sent over the wire by an action


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
        # self.pixel_desc = {}
        self.map_width = 0
        self.map_height = 0
        self.robot_pose = None
        self.received_choices = {}
        self.failed_points = []
        self.move_attempt = 0
        self.coverage_ratio = 0.0
        self.current_point = None
        self.robot_ranges = []
        self.publisher_map = {}
        self.close_devices = []
        self.is_exploring = False
        self.moving_to_frontier = False
        self.start_time = 0
        self.prev_pose = 0
        rospy.init_node("gvg_explore", anonymous=True)
        self.robot_id = rospy.get_param('~robot_id')
        self.run = rospy.get_param("~run")
        self.min_edge_length = rospy.get_param("/robot_{}/min_edge_length".format(self.robot_id)) * SCALE
        self.min_hallway_width = rospy.get_param("/robot_{}/min_hallway_width".format(self.robot_id)) * SCALE
        self.comm_range = rospy.get_param("/robot_{}/comm_range".format(self.robot_id)) * SCALE
        self.point_precision = rospy.get_param("/robot_{}/point_precision".format(self.robot_id))
        self.lidar_scan_radius = rospy.get_param("/robot_{}/lidar_scan_radius".format(self.robot_id)) * SCALE
        self.info_scan_radius = rospy.get_param("/robot_{}/info_scan_radius".format(self.robot_id)) * SCALE
        self.lidar_fov = rospy.get_param("/robot_{}/lidar_fov".format(self.robot_id))
        self.slope_bias = rospy.get_param("/robot_{}/slope_bias".format(self.robot_id)) * SCALE
        self.separation_bias = rospy.get_param("/robot_{}/separation_bias".format(self.robot_id)) * SCALE
        self.opposite_vector_bias = rospy.get_param("/robot_{}/opposite_vector_bias".format(self.robot_id)) * SCALE
        self.robot_count = rospy.get_param("~robot_count")
        self.environment = rospy.get_param("~environment")
        rospy.Subscriber("/robot_{}/MoveTo/status".format(self.robot_id), GoalStatusArray, self.move_status_callback)
        rospy.Subscriber("/robot_{}/MoveTo/result".format(self.robot_id), MoveToPosition2DActionResult,
                         self.move_result_callback)
        rospy.Subscriber("/robot_{}/navigator/plan".format(self.robot_id), GridCells, self.navigation_plan_callback)
        rospy.Subscriber("/robot_{}/edge_list".format(self.robot_id), EdgeList, self.edge_list_callback)
        self.move_to_stop = rospy.ServiceProxy('/robot_{}/Stop'.format(self.robot_id), Trigger)
        self.moveTo_pub = rospy.Publisher("/robot_{}/MoveTo/goal".format(self.robot_id), MoveToPosition2DActionGoal,
                                          queue_size=10)
        self.pose_publisher = rospy.Publisher("/robot_{}/cmd_vel".format(self.robot_id), Twist, queue_size=1)
        rospy.Subscriber("/robot_{}/odom".format(self.robot_id), Odometry, callback=self.pose_callback)
        rospy.Subscriber("/chosen_point", ChosenPoint, self.chosen_point_callback)
        self.chose_point_pub = rospy.Publisher("/chosen_point", ChosenPoint, queue_size=1000)
        rospy.Subscriber('/robot_{}/coverage'.format(self.robot_id), Coverage, self.coverage_callback)
        self.exploration_feedback = GvgExploreFeedback()
        self.exploration_result = GvgExploreResult()
        self.action_server = actionlib.SimpleActionServer('/robot_{}/gvg_explore'.format(self.robot_id),
                                                          GvgExploreAction, execute_cb=self.start_gvg_exploration,
                                                          auto_start=False)
        self.action_server.start()
        self.listener = tf.TransformListener()
        rospy.loginfo("Robot {}: Exploration server online...".format(self.robot_id))
        self.spin()

    def spin(self):
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            r.sleep()

    def edge_list_callback(self, data):
        # if not self.is_exploring:
        #     rospy.logerr("New edges received")
        edgelist = data.edges
        self.edges.clear()
        for r in edgelist:
            edge = self.get_edge(r)
            self.edges.update(edge)
        self.create_adjlist()

    def start_gvg_exploration(self, goal):
        success = True
        visited_nodes = {}
        robot_pose = (goal.pose.position.x, goal.pose.position.y, goal.pose.position.z)
        self.moving_to_frontier = True
        self.move_robot_to_goal(robot_pose)
        rospy.logerr("Robot {} going to frontier....".format(self.robot_id))
        while self.moving_to_frontier:
            continue
        rospy.logerr("Robot {} arrived at frontier. Starting Exploration now...".format(self.robot_id))
        while self.coverage_ratio < 0.8:
            try:
                rospy.loginfo('gvg exploration: Recomputing frontier')
                self.is_exploring = True
                self.run_dfs(visited_nodes)
                self.is_exploring = False
            except Exception as e:
                rospy.logerr(e.message)

        if success:
            self.create_result(visited_nodes)
            rospy.loginfo('gvg exploration: Succeeded')
            self.action_server.set_succeeded(self.exploration_result)

    def run_dfs(self, visited_nodes):
        robot_pose = self.get_robot_pose()
        pose = scale_up(robot_pose)
        edge = self.get_closest_ridge(pose, visited_nodes)
        leaf = edge[1]
        leaf_parent = edge[0]
        S = [leaf]
        parents = {leaf: leaf_parent}
        # --------- DFS -------
        rospy.logerr("Robot {}: Starting DFS..".format(self.robot_id))
        while len(S) > 0:
            u = S.pop()
            unvisited_ans = [u]
            # for plotting
            if self.robot_id == 0:
                pose = self.get_robot_pose()
                point = scale_up(pose)
                self.plot_intersections(point, u)
            # for plotting ends here
            self.explore_point(u)
            visited_nodes[u] = None
            self.bfs(unvisited_ans, visited_nodes)
            child_leaf = self.get_child_leaf(u, parents[u], visited_nodes)
            if not child_leaf:
                new_robot_pose = self.get_robot_pose()
                new_pose = scale_up(new_robot_pose)
                child_leaf = self.get_closest_ridge(new_pose, visited_nodes, is_initial=True)
            if child_leaf:
                S.append(child_leaf[1])
                parents[child_leaf[1]] = child_leaf[0]

    def get_closest_ridge(self, robot_pose, all_visited, is_initial=False):
        close_edge = None
        closest_ridge = {}
        edge_list = list(self.edges)
        vertex_dict = {}
        for e in edge_list:
            p1 = e[0]
            p2 = e[1]
            if not self.is_visited(p1, all_visited) or not self.is_visited(p2, all_visited):
                o = self.edges[e]
                width = D(o[0], o[1])
                v1 = get_vector(p1, p2)
                desc = (v1, width)
                vertex_dict[e] = desc
                d = min([D(robot_pose, e[0]), D(robot_pose, e[1])])
                closest_ridge[e] = d
        if closest_ridge:
            if is_initial:
                edge = max(closest_ridge, key=closest_ridge.get)
            else:
                edge = min(closest_ridge, key=closest_ridge.get)
            close_edge = self.get_child_leaf(edge[1], edge[0], all_visited)

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
            close_edge = min(closest_leaf_edge, key=closest_leaf_edge.get)

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

    def coverage_callback(self, data):
        self.coverage_ratio = data.coverage

    def create_feedback(self, point):
        next_point = scale_down(point)  # pixel2pose(point, self.origin_x, self.origin_y,self.resolution)
        self.exploration_feedback.progress = "String Exploring"
        pose = Pose()
        pose.position.x = next_point[INDEX_FOR_X]
        pose.position.y = next_point[INDEX_FOR_Y]
        self.exploration_feedback.current_point = pose

    def create_result(self, visited_nodes):
        all_nodes = list(visited_nodes)
        allposes = []
        for n in all_nodes:
            point = scale_down(n)  # pixel2pose(n, self.origin_x, self.origin_y,self.resolution)
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
        # v_nodes = list(visited)
        # for n in v_nodes:
        #     if D(n, v) < self.lidar_scan_radius:
        #         visited[v] = None
        #         already_visited = True
        #         break
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
                    local_leaves.append((u, nei))
                local_visited.append(u)
        dists = {}
        for l in local_leaves:
            dists[D(first_node, l[1])] = l
        if dists:
            next_leaf = dists[min(dists.keys())]
        else:
            other_dists = {}
            for k, v in self.leaves.items():
                if not self.is_visited(k[1], all_visited):
                    other_dists[D(first_node, k[1])] = k
            if other_dists:
                next_leaf = other_dists[min(other_dists.keys())]
        return next_leaf

    def bfs(self, aux_nodes, visited_nodes):
        pose = self.get_robot_pose()
        p_pixel = scale_up(pose)
        H = []
        for p in aux_nodes:
            if not self.is_visited(p, visited_nodes):
                heapq.heappush(H, (D(p_pixel, p), p))
        if len(H) == 0:
            return
        n = heapq.heappop(H)
        self.explore_point(n[1])
        aux_nodes.remove(n[1])
        visited_nodes[n[1]] = None
        self.bfs(aux_nodes, visited_nodes)

    def explore_point(self, goal_pose):
        explored_pose = scale_down(goal_pose)
        self.has_arrived = False
        self.move_attempt = 0
        self.move_robot_to_goal(explored_pose)
        # rospy.logerr("Explored point {}".format(explored_pose))
        while not self.has_arrived and not self.navigation_failed:
            # rospy.logerr("Robot exploring")
            continue
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
        # rospy.logerr("Going to: {}".format(self.current_point))
        id_val = "robot_{}_{}_explore".format(self.robot_id, self.goal_count)
        move = MoveToPosition2DActionGoal()
        move.header.frame_id = '/robot_{}/map'.format(self.robot_id)
        goal_id = GoalID()
        goal_id.id = id_val
        move.goal_id = goal_id
        move.goal.target_pose.x = goal[INDEX_FOR_X]
        move.goal.target_pose.y = goal[INDEX_FOR_Y]
        self.moveTo_pub.publish(move)
        self.prev_pose = self.get_robot_pose()
        self.start_time = rospy.Time.now().secs
        chosen_point = ChosenPoint()
        chosen_point.header.frame_id = '{}'.format(self.robot_id)
        chosen_point.x = goal[0]
        chosen_point.y = goal[1]
        chosen_point.direction = direction
        self.chose_point_pub.publish(chosen_point)
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
                    # rospy.logerr("Robot type, {} Heading to Frontier point...{}".format(self.robot_id,goal_status.status))
                    if goal_status.status == ACTIVE:
                        now = rospy.Time.now().secs
                        if (now - self.start_time) > 5:
                            pose = self.get_robot_pose()
                            if D(self.prev_pose, pose) < 1.5:
                                self.has_arrived = True
                                # self.move_to_stop()
                            self.prev_pose = pose
                            self.start_time = now
                    else:
                        if not self.has_arrived:
                            # rospy.logerr("Robot type, {} Heading to Frontier point...{}".format(self.robot_id,goal_status.status))
                            # self.move_to_stop()
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
                    # rospy.logerr("Robot {} can't reach goal..".format(self.robot_id))
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

    def chosen_point_callback(self, data):
        self.received_choices[(data.x, data.y)] = data

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

    def plot_intersections(self, robot_pose, next_leaf):
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
        plt.grid()
        plt.savefig("gvg/current_leaves_{}_{}_{}.png".format(self.robot_id, time.time(), self.run))
        plt.close(fig)
        # plt.show()


if __name__ == "__main__":
    graph = GVGExplore()
