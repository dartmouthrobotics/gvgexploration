#!/usr/bin/python

import time

import rospy
from std_msgs.msg import *
from std_srvs.srv import *
from nav2d_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from geometry_msgs.msg import *
import math
from graph import Graph
from gvgexploration.msg import *
from gvgexploration.srv import *
from nav2d_navigator.msg import MoveToPosition2DActionGoal, MoveToPosition2DActionResult
from actionlib_msgs.msg import GoalStatusArray, GoalID
import numpy as np
from time import sleep
from threading import Thread, Lock
import copy
from os import path
import tf
import pickle
import os
from os.path import exists
import project_utils as pu
from actionlib import SimpleActionClient

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
TURNING_ANGLE = np.deg2rad(45)


class Robot:
    def __init__(self, robot_id, robot_type=0, base_stations=[], relay_robots=[], frontier_robots=[]):
        self.lock = Lock()
        self.robot_id = robot_id
        self.data_file_name = 'scan_data_{}.pickle'.format(self.robot_id)
        self.robot_type = robot_type
        self.base_stations = base_stations
        self.relay_robots = relay_robots
        self.frontier_robots = frontier_robots
        self.candidate_robots = []
        self.auction_publishers = {}
        self.frontier_point = None
        self.robot_state = ACTIVE_STATE
        self.publisher_map = {}
        self.allocation_pub = {}
        self.auction_feedback_pub = {}
        self.initial_data_count = 0
        self.is_exploring = False
        self.exploration_started = False
        self.karto_messages = {}
        self.start_exploration = None
        self.received_choices = {}
        self.robot_pose = None
        self.signal_strength = {}
        self.close_devices = []
        self.received_devices = 0
        self.is_initial_sharing = True
        self.is_sender = False
        self.previous_pose = None
        self.is_initial_data_sharing = True
        self.initial_receipt = True
        self.is_initial_move = True
        self.exploration_data_senders = {}
        self.last_map_update_time = rospy.Time.now().secs
        self.frontier_points = []
        self.coverage_ratio = 0.0
        self.goal_handle = None
        self.auction_feedback = {}
        self.all_feedbacks = {}
        self.in_charge_of_computation = False
        self.waiting_for_response = False

        self.candidate_robots = self.frontier_robots + self.base_stations
        self.rate = rospy.Rate(0.1)
        self.run = rospy.get_param("~run")
        for rid in self.candidate_robots:
            pub = rospy.Publisher("/robot_{}/received_data".format(rid), BufferedData, queue_size=1000)
            pub1 = rospy.Publisher("/robot_{}/auction_points".format(rid), Auction, queue_size=10)
            pub2 = rospy.Publisher("/robot_{}/allocated_point".format(rid), Frontier, queue_size=10)
            pub3 = rospy.Publisher("/robot_{}/auction_feedback".format(rid), Auction, queue_size=10)
            self.allocation_pub[rid] = pub2
            self.publisher_map[rid] = pub
            self.auction_publishers[rid] = pub1
            self.auction_feedback_pub[rid] = pub3

        self.karto_pub = rospy.Publisher("/robot_{}/karto_in".format(self.robot_id), LocalizedScan, queue_size=1000)
        rospy.Subscriber('/roscbt/robot_{}/received_data'.format(self.robot_id), BufferedData,
                         self.buffered_data_callback, queue_size=100)
        rospy.Subscriber("/roscbt/robot_{}/signal_strength".format(self.robot_id), SignalStrength,
                         self.signal_strength_callback)
        rospy.Subscriber('/karto_out'.format(self.robot_id), LocalizedScan, self.robots_karto_out_callback,
                         queue_size=1000)
        rospy.Subscriber("/robot_{}/auction_points".format(self.robot_id), Auction, self.auction_points_callback)
        self.fetch_frontier_points = rospy.ServiceProxy('/robot_{}/frontier_points'.format(self.robot_id),
                                                        FrontierPoint)
        self.check_intersections = rospy.ServiceProxy('/robot_{}/check_intersections'.format(self.robot_id),
                                                      Intersections)
        rospy.Subscriber('/robot_{}/coverage'.format(self.robot_id), Coverage, self.coverage_callback)
        rospy.Subscriber('/robot_{}/allocated_point'.format(self.robot_id), Frontier, self.allocated_point_callback)
        rospy.Subscriber('/robot_{}/auction_feedback'.format(self.robot_id), Auction, self.auction_feedback_callback)
        rospy.Subscriber('/robot_{}/map'.format(self.robot_id), OccupancyGrid, self.map_update_callback)

        # robot identification sensor
        self.robot_range_pub = rospy.Publisher("/robot_{}/robot_ranges".format(self.robot_id), RobotRange, queue_size=5)
        self.robot_poses = {}

        self.trans_matrices = {}
        self.inverse_trans_matrices = {}
        self.prev_intersects = []
        all_robots = self.candidate_robots + [self.robot_id]
        for i in all_robots:
            s = "def a_" + str(i) + "(self, data): self.robot_poses[" + str(i) + "] = (data.pose.pose.position.x," \
                                                                                 "data.pose.pose.position.y," \
                                                                                 "(data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,data.pose.pose.orientation.w)) "
            exec (s)
            exec ("setattr(Robot, 'callback_pos_teammate" + str(i) + "', a_" + str(i) + ")")
            exec ("rospy.Subscriber('/robot_" + str(
                i) + "/base_pose_ground_truth', Odometry, self.callback_pos_teammate" + str(i) + ", queue_size = 100)")

        # ======= pose transformations====================
        self.listener = tf.TransformListener()
        self.exploration = SimpleActionClient("/robot_{}/gvg_explore".format(self.robot_id), GvgExploreAction)
        self.exploration.wait_for_server()
        rospy.loginfo("Robot {} Initialized successfully!!".format(self.robot_id))

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            if self.coverage_ratio > MAX_COVERAGE_RATIO:
                self.cancel_exploration()
                rospy.logerr("Robot {} Coverage: {}".format(self.robot_id, self.coverage_ratio))
                rospy.is_shutdown()  # experiment done
                break
            else:
                # try:
                self.publish_robot_ranges()
                if self.exploration_started:
                    self.check_data_sharing_status()
                r.sleep()
            # except Exception as e:
            #     rospy.logerr('Error: {}'.format(e))
            #     break

    def check_data_sharing_status(self):
        if not self.waiting_for_response:
            robot_pose = self.get_robot_pose()
            poses = list(self.robot_poses.values())
            actual_poses = []
            for p in poses:
                ap = Pose()
                ap.position.x = p[0]
                ap.position.y = p[1]
                actual_poses.append(ap)
            response = self.check_intersections(IntersectionsRequest(poses=actual_poses))
            time_stamp = rospy.Time.now().secs
            # same_as_previous = all([p for p in intersections if p not in self.prev_intersects])
            # if P and intersections:
            if response.result:
                if self.close_devices:  # and not same_as_previous:
                    rospy.logerr("Robot {} sending data to Robot {}".format(self.robot_id, self.close_devices))
                    self.is_sender = True
                    self.waiting_for_response = True
                    self.push_messages_to_receiver(self.close_devices)
                    self.save_data([{'time': time_stamp, 'pose': robot_pose, 'intersections': []}],
                                   'gvg/interconnections_{}_{}.pickle'.format(self.robot_id, self.run))

            self.prev_intersects = []

    def map_update_callback(self, data):
        self.last_map_update_time = rospy.Time.now().secs

    def wait_for_updates(self):
        while (rospy.Time.now().secs - self.last_map_update_time) < 5:
            continue

    def coverage_callback(self, data):
        self.coverage_ratio = data.coverage

    def allocated_point_callback(self, data):
        rospy.logerr("Robot {}: Received frontier location from {}".format(self.robot_id, data.header.frame_id))
        self.frontier_point = data.pose
        robot_pose = self.get_robot_pose()
        new_point = (data.pose.position.x, data.pose.position.y, 0)
        self.save_data([{'time': rospy.Time.now().secs, 'pose': robot_pose, 'frontier': new_point}],
                       'gvg/frontiers_{}_{}.pickle'.format(self.robot_id, self.run))
        if self.is_initial_data_sharing:
            self.is_initial_data_sharing = False

        self.is_exploring = True
        self.robot_state = ACTIVE_STATE
        self.start_exploration_action(self.frontier_point)

    def auction_feedback_callback(self, data):
        global robot_distances
        rospy.logerr("Robot {}: Received bids from {}".format(self.robot_id, data.header.frame_id))
        self.all_feedbacks[data.header.frame_id] = data
        rid = data.header.frame_id
        m = len(data.distances)
        min_dist = max(data.distances)
        min_pose = None
        for i in range(m):
            if min_dist >= data.distances[i]:
                min_pose = data.poses[i]
                min_dist = data.distances[i]
        self.auction_feedback[rid] = (min_dist, min_pose)
        if len(self.all_feedbacks) == len(self.close_devices):
            all_robots = list(self.auction_feedback)
            taken_poses = []
            rospy.logerr("Computing poses for : {}".format(all_robots))
            for i in all_robots:
                pose_i = (self.auction_feedback[i][1].position.x, self.auction_feedback[i][1].position.y)
                conflicts = []
                for j in all_robots:
                    if i != j:
                        pose_j = (self.auction_feedback[j][1].position.x, self.auction_feedback[j][1].position.y)
                        if pose_i == pose_j:
                            conflicts.append((i, j))
                rpose = self.auction_feedback[i][1]
                frontier = Frontier()
                frontier.header.frame_id = str(self.robot_id)
                frontier.pose = rpose
                self.allocation_pub[i].publish(frontier)

                taken_poses.append(pose_i)
                for c in conflicts:
                    conflicting_robot_id = c[1]
                    feedback_for_j = self.all_feedbacks[conflicting_robot_id]
                    rob_poses = feedback_for_j.poses
                    rob_distances = feedback_for_j.distances
                    remaining_poses = {}
                    for k in range(len(rob_poses)):
                        p = (rob_poses[k].position.x, rob_poses[k].position.y)
                        if p != pose_i and p not in taken_poses:
                            remaining_poses[rob_distances[k]] = p
                    next_closest_dist = min(list(remaining_poses))
                    self.auction_feedback[conflicting_robot_id] = (
                        next_closest_dist, remaining_poses[next_closest_dist])
            # now find where to go
            robot_pose = self.get_robot_pose()
            self.frontier_point = None
            dist_dict = {}
            for point in self.frontier_points:
                dist_dict[pu.D(robot_pose, point)] = point

            while not self.frontier_point and dist_dict:
                min_dist = min(list(dist_dict))
                closest = dist_dict[min_dist]
                if (closest[0], closest[1]) not in taken_poses:
                    pose = Pose()
                    pose.position.x = closest[0]
                    pose.position.y = closest[1]
                    self.frontier_point = pose
                    break
                del dist_dict[min_dist]
            if self.frontier_point:
                self.start_exploration_action(self.frontier_point)
            else:
                rospy.logerr("Robot {}: No frontier point left! ".format(self.robot_id))
            self.all_feedbacks.clear()
            self.auction_feedback.clear()
            self.exploration_data_senders.clear()
            self.waiting_for_response = False
        else:
            rospy.logerr('Robot {}: Waiting for more bids: {}, {}'.format(self.robot_id, self.close_devices,
                                                                          self.auction_feedback))

    def auction_points_callback(self, data):
        rospy.logerr("Robot {} auction points received from {}".format(self.robot_id, data.header.frame_id))
        sender_id = data.header.frame_id
        poses = data.poses
        received_points = []
        distances = []
        robot_pose = self.get_robot_pose()
        for p in poses:
            received_points.append(p)
            point = (p.position.x, p.position.y,
                     self.get_elevation((p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)))
            distance = pu.D(robot_pose, point)
            distances.append(distance)
        auction = Auction()
        auction.header.frame_id = str(self.robot_id)
        auction.poses = received_points
        auction.distances = distances
        self.auction_feedback_pub[sender_id].publish(auction)
        rospy.logerr("Robot {} auction bids sent back to {}".format(self.robot_id, sender_id))

    def publish_auction_points(self, rendezvous_poses, receivers, direction=1, total_exploration_time=0):
        x = [float(v[0]) for v in rendezvous_poses]
        y = [float(v[1]) for v in rendezvous_poses]
        auction = Auction()
        auction.header.frame_id = str(self.robot_id)
        auction.header.stamp = rospy.Time.now()
        auction.poses = []
        self.auction_feedback.clear()
        self.all_feedbacks.clear()
        for i in range(len(x)):
            pose = Pose()
            pose.position.x = x[i]
            pose.position.y = y[i]
            auction.poses.append(pose)
        for id in receivers:
            self.auction_publishers[id].publish(auction)

    def robots_karto_out_callback(self, data):
        if data.robot_id - 1 == self.robot_id:
            for rid in self.candidate_robots:
                self.add_to_file(rid, [data])
            if self.is_initial_data_sharing:
                self.push_messages_to_receiver(self.candidate_robots)
                self.is_initial_data_sharing = False

    def push_messages_to_receiver(self, receiver_ids, is_alert=0):
        for receiver_id in receiver_ids:
            message_data = self.load_data_for_id(receiver_id)
            buffered_data = BufferedData()
            buffered_data.header.frame_id = '{}'.format(self.robot_id)
            buffered_data.header.stamp.secs = rospy.Time.now().secs
            buffered_data.secs = []
            buffered_data.data = message_data
            buffered_data.alert_flag = is_alert
            self.publisher_map[str(receiver_id)].publish(buffered_data)
            self.delete_data_for_id(receiver_id)

    def signal_strength_callback(self, data):
        signals = data.signals
        time_stamp = data.header.stamp.secs
        robots = []
        devices = []
        for rs in signals:
            robots.append([rs.robot_id, rs.rssi])
            devices.append(str(rs.robot_id))
        self.close_devices = devices

    def process_data(self, sender_id, buff_data):
        data_vals = buff_data.data
        counter = 0
        r = rospy.Rate(1)
        for scan in data_vals:
            self.karto_pub.publish(scan)
            counter += 1
            r.sleep()
        for rid in self.candidate_robots:
            if rid != sender_id:
                self.add_to_file(rid, data_vals)

    def buffered_data_callback(self, buff_data):
        sender_id = buff_data.header.frame_id
        if self.is_exploring:
            self.exploration_data_senders[sender_id] = None
        self.process_data(sender_id, buff_data)
        if self.is_initial_sharing:
            self.wait_for_updates()
        if self.initial_receipt:
            rospy.logerr(
                "Robot {} initial data from {}: {} files".format(self.robot_id, sender_id, len(buff_data.data)))
            self.initial_data_count += 1
            if self.initial_data_count == len(self.candidate_robots):
                self.initial_receipt = False
                if self.i_have_least_id():
                    rospy.logerr("Robot {}: Computing frontier points...".format(self.robot_id))
                    frontier_point_response = self.fetch_frontier_points(
                        FrontierPointRequest(count=len(self.candidate_robots) + 1))
                    frontier_points = self.parse_frontier_response(frontier_point_response)
                    if frontier_points:
                        rospy.logerr('Robot {}: Received frontier points: {}'.format(self.robot_id, frontier_points))
                        self.frontier_points = frontier_points
                        self.publish_auction_points(frontier_points, self.candidate_robots, direction=TO_FRONTIER)
                    else:
                        rospy.logerr("Robot {}: No valid frontier points".format(self.robot_id))
                else:
                    rospy.logerr("Robot {}: Waiting for frontier points...".format(self.robot_id))
        else:
            rospy.logerr(
                "Robot {} received data from {}: {} files".format(self.robot_id, sender_id, len(buff_data.data)))
            if not self.is_sender:
                rospy.logerr(
                    "Robot {} exploration data {}: {} files".format(self.robot_id, sender_id, len(buff_data.data)))
                self.push_messages_to_receiver([sender_id])
                self.cancel_exploration()
            else:
                rospy.logerr("Robot {} exploration response from {}: {} files".format(self.robot_id, sender_id,
                                                                                      len(buff_data.data)))
                self.received_devices += 1
                if self.received_devices == len(self.close_devices):
                    self.received_devices = 0
                    self.is_sender = False
                    robot_pose = self.get_robot_pose()
                    # if self.robot_id < int(sender_id):
                    rospy.logerr("Robot {} Recomputing frontier points".format(self.robot_id))
                    self.cancel_exploration()
                    frontier_point_response = self.fetch_frontier_points(FrontierPointRequest(count=len(self.candidate_robots) + 1))
                    frontier_points = self.parse_frontier_response(frontier_point_response)
                    rospy.logerr('Robot {}: Received frontier points: {}'.format(self.robot_id, frontier_points))
                    if frontier_points:
                        self.frontier_points = frontier_points
                        self.publish_auction_points(frontier_points, self.exploration_data_senders,
                                                    direction=TO_FRONTIER)
                    else:
                        rospy.logerr("Robot {}: No valid frontier points".format(self.robot_id))


    def start_exploration_action(self, pose):
        goal = GvgExploreGoal(pose=pose)
        self.exploration.wait_for_server()
        self.goal_handle = self.exploration.send_goal(goal)
        self.exploration_started = True
        self.exploration.wait_for_result()
        self.exploration.get_result()  # A

    def parse_frontier_response(self, data):
        frontier_points = []
        received_poses = data.poses
        if received_poses:
            for p in received_poses:
                yaw = self.get_elevation((p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))
                frontier_points.append((p.position.x, p.position.y, yaw))
        return frontier_points

    def chosen_point_callback(self, data):
        self.received_choices[(data.x, data.y)] = data

    def get_available_points(self, points):
        available_points = []
        for p in points:
            if p not in self.received_choices:
                if len(self.received_choices):
                    for point in self.received_choices:
                        distance = math.sqrt((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2)
                        if distance > ROBOT_SPACE:
                            available_points.append(p)
                else:
                    available_points.append(p)
        return available_points

    def get_closest_point(self, robot_pose, optimal_points):
        distance = {}
        closest_point = None
        for p in optimal_points:
            distance[p] = self.D(robot_pose, p)
        if distance:
            closest_point = min(distance, key=distance.get)
        return closest_point

    def i_have_least_id(self):
        id = int(self.robot_id)
        least = min([int(i) for i in self.candidate_robots])
        return id < least

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
                # rospy.logerr("Robot {}: Can't fetch robot pose from tf".format(self.robot_id))
                pass

        return robot_pose

    def cancel_exploration(self):
        if self.goal_handle:
            self.goal_handle.cancel()
            rospy.logerr("Robot {} Canceling exploration ...".format(self.robot_id))

    def add_to_file(self, rid, data):
        # self.lock.acquire()
        if rid in self.karto_messages:
            self.karto_messages[rid] += data
        else:
            self.karto_messages[rid] = data
        # self.lock.release()
        return True

    def load_data_for_id(self, rid):
        message_data = []
        if rid in self.karto_messages:
            message_data = self.karto_messages[rid]

        return message_data

    def delete_data_for_id(self, rid):
        self.lock.acquire()
        if rid in self.karto_messages:
            del self.karto_messages[rid]
        self.lock.release()
        return True

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
                    rospy.logerr("error saving data: {}".format(e))
        return data_dict

    def theta(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        if dx == 0:
            dx = 0.000001
        return math.atan2(dy, dx)

    def D(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def publish_robot_ranges(self):
        if len(self.robot_poses) == len(self.candidate_robots) + 1:
            self.generate_transformation_matrices()
            if len(self.trans_matrices) == len(self.candidate_robots) + 1:
                pose = self.robot_poses[self.robot_id]
                yaw = self.get_elevation(pose[2])
                robotV = np.asarray([pose[0], pose[1], yaw, 1])

                distances = []
                angles = []
                for rid in self.candidate_robots:
                    robot_c = self.get_robot_coordinate(rid, robotV)
                    distance = self.D(pose, robot_c)
                    angle = self.theta(pose, robot_c)
                    distances.append(distance)
                    angles.append(angle)
                if distances != []:
                    robot_range = RobotRange()
                    robot_range.distances = distances
                    robot_range.angles = angles
                    robot_range.robot_id = self.robot_id
                    robot_range.header.stamp = rospy.Time.now()
                    self.robot_range_pub.publish(robot_range)

    def get_elevation(self, quaternion):
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        return yaw

    def generate_transformation_matrices(self):
        map_origin = (0, 0, 0, 1)
        for rid in self.candidate_robots + [self.robot_id]:
            p = self.robot_poses[int(rid)]
            theta = self.theta(map_origin, p)
            M = [[np.cos(theta), -1 * np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
            T = copy.deepcopy(M)
            T.append([0, 0, 0, 1])
            yaw = self.get_elevation(p[2])
            T[0].append(p[0])
            T[1].append(p[1])
            T[2].append(yaw)
            tT = np.asarray(T)
            self.trans_matrices[rid] = tT
            tT_inv = np.linalg.inv(tT)
            self.inverse_trans_matrices[int(rid)] = tT_inv

    def get_robot_coordinate(self, rid, V):
        other_T = self.trans_matrices[rid]  # oTr2
        robot_T = self.inverse_trans_matrices[self.robot_id]  #
        cTr = other_T.dot(robot_T)
        other_robot_pose = cTr.dot(V)
        return other_robot_pose.tolist()


if __name__ == "__main__":
    # initializing a node
    rospy.init_node("robot_node", anonymous=True)
    robot_id = int(rospy.get_param("~robot_id", 0))
    robot_type = int(rospy.get_param("~robot_type", 0))
    base_stations = str(rospy.get_param("~base_stations", ''))
    relay_robots = str(rospy.get_param("~relay_robots", ''))
    frontier_robots = str(rospy.get_param("~frontier_robots", ''))

    relay_robots = relay_robots.split(',')
    frontier_robots = frontier_robots.split(',')
    base_stations = base_stations.split(',')

    rospy.loginfo("ROBOT_ID: {},RELAY ROBOTS: {}, FRONTIER ROBOTS: {}, BASE STATION: {}".format(robot_id, relay_robots,
                                                                                                frontier_robots,
                                                                                                base_stations))
    if relay_robots[0] == '':
        relay_robots = []
    if frontier_robots[0] == '':
        frontier_robots = []
    if base_stations[0] == '':
        base_stations = []
    robot = Robot(robot_id, robot_type=robot_type, base_stations=base_stations, relay_robots=relay_robots,
                  frontier_robots=frontier_robots)
    robot.spin()
