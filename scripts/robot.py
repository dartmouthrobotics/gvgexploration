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
from nav2d_navigator.msg import MoveToPosition2DActionGoal, MoveToPosition2DActionResult
from actionlib_msgs.msg import GoalStatusArray, GoalID
from std_srvs.srv import Trigger

from time import sleep
from threading import Thread, Lock
import copy
from os import path
import tf
import pickle
import os
from os.path import exists

MIN_RANGE = 1
SLOPE_MARGIN = 1.0
INF = 1000000000000
NEG_INF = -1000000000000
# ids to identify the robot type
BS_TYPE = 1
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

# server

START_SCAN = '1'
STOP_SCAN = '0'
MAX_ATTEMPTS = 2
MIN_SIGNAL_STRENGTH = -1 * 65.0


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
        self.rendezvous_publishers = {}
        self.frontier_point = None
        self.previous_frontier_point = None
        self.graph_processor = Graph(robot_id)
        self.robot_state = ACTIVE_STATE
        self.publisher_map = {}
        self.initial_data_count = 0
        self.is_exploring = False
        self.exploration_id = None
        self.karto_messages = {}
        self.last_map_update_time = rospy.Time.now().secs
        self.last_buffered_data_time = rospy.Time.now().secs
        self.goal_count = 0
        self.exploration_start_time = None
        self.total_exploration_time = None
        self.moveTo_pub = None
        self.start_exploration = None
        self.move_to_stop = None
        self.cancel_explore_pub = None
        self.received_choices = {}
        self.vertex_descriptions = {}
        self.navigation_plan = None
        self.waiting_for_plan = False
        self.start_now = None
        self.robot_pose = None
        self.signal_strength = {}
        self.close_devices = []
        self.received_devices = 0
        self.move_attempt = 0
        self.is_initial_rendezvous_sharing = True
        self.received_choices = {}
        self.is_sender = False
        self.previous_pose = None
        self.prev_slope = -1
        self.is_initialization = True
        self.map_update_is_active = False
        self.others_active = False
        self.map_update_count = 0
        self.is_initial_data_sharing = True
        self.initial_receipt = True
        self.candidate_robots = self.frontier_robots + self.base_stations
        self.rate = rospy.Rate(0.1)
        for rid in self.candidate_robots:
            pub = rospy.Publisher("/robot_{}/received_data".format(rid), BufferedData, queue_size=1000)
            pub1 = rospy.Publisher("/robot_{}/rendezvous_points".format(rid), RendezvousPoints, queue_size=10)
            self.publisher_map[rid] = pub
            self.rendezvous_publishers[rid] = pub1
            rospy.Subscriber("/robot_{}/start_exploration".format(rid), String, self.start_exploration_callback)

        self.karto_pub = rospy.Publisher("/robot_{}/karto_in".format(self.robot_id), LocalizedScan, queue_size=1000)
        self.chose_point_pub = rospy.Publisher("/chosen_point", ChosenPoint, queue_size=1000)
        self.auction_point_pub = rospy.Publisher("/auction_point", ChosenPoint, queue_size=1000)
        self.start_exploration_pub = rospy.Publisher("/robot_{}/start_exploration".format(self.robot_id), String,
                                                     queue_size=10)
        self.moveTo_pub = rospy.Publisher("/robot_{}/MoveTo/goal".format(self.robot_id), MoveToPosition2DActionGoal,
                                          queue_size=10)
        self.chose_point_pub = rospy.Publisher("/chosen_point", ChosenPoint, queue_size=1000)
        self.cancel_explore_pub = rospy.Publisher("/robot_{}/Explore/cancel".format(self.robot_id), GoalID,
                                                  queue_size=10)
        self.pose_publisher = rospy.Publisher("/robot_{}/cmd_vel".format(self.robot_id), Twist, queue_size=1)
        self.start_exploration = rospy.ServiceProxy('/robot_{}/StartExploration'.format(self.robot_id), Trigger)
        self.move_to_stop = rospy.ServiceProxy('/robot_{}/Stop'.format(self.robot_id), Trigger)

        rospy.Subscriber('/roscbt/robot_{}/received_data'.format(self.robot_id), BufferedData,
                         self.buffered_data_callback, queue_size=100)
        rospy.Subscriber('/roscbt/robot_{}/rendezvous_points'.format(self.robot_id), RendezvousPoints,
                         self.callback_rendezvous_points)
        rospy.Subscriber("/roscbt/robot_{}/signal_strength".format(self.robot_id), SignalStrength,
                         self.signal_strength_callback)
        rospy.Subscriber("/robot_{}/map".format(self.robot_id), OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber("/robot_{}/MoveTo/status".format(self.robot_id), GoalStatusArray, self.move_status_callback)
        rospy.Subscriber("/robot_{}/MoveTo/result".format(self.robot_id), MoveToPosition2DActionResult,
                         self.move_result_callback)
        rospy.Subscriber('/robot_{}/Explore/status'.format(self.robot_id), GoalStatusArray, self.exploration_callback)
        rospy.Subscriber("/robot_{}/navigator/plan".format(self.robot_id), GridCells, self.navigation_plan_callback)
        rospy.Subscriber("/chosen_point", ChosenPoint, self.chosen_point_callback)
        rospy.Subscriber('/karto_out', LocalizedScan, self.robots_karto_out_callback, queue_size=1000)
        rospy.Subscriber("/chosen_point", ChosenPoint, self.chosen_point_callback)
        rospy.Subscriber("/robot_{}/pose".format(self.robot_id), Pose, callback=self.pose_callback)

        # ======= pose transformations====================
        self.listener = tf.TransformListener()
        rospy.loginfo("Robot {} Initialized successfully!!".format(self.robot_id))

    def spin(self):
        r = rospy.Rate(0.1)
        self.clear_data()
        while not rospy.is_shutdown():
            # try:
            if self.is_exploring:
                self.direction_has_changed()
            r.sleep()
        # except Exception as e:
        #     rospy.logerr('Error: {}'.format(e))
        #     break

    def callback_rendezvous_points(self, data):
        # if int(data.header.frame_id) < int(self.robot_id):
        rospy.logerr("Robot {} received points from {}".format(self.robot_id, data.header.frame_id))
        x = data.x
        y = data.y
        direction = data.direction
        received_points = [(x[i], y[i]) for i in range(len(x))]
        self.wait_for_map_update()

        robot_pose = self.get_robot_pose()
        valid_points = self.get_available_points(received_points)
        new_point = self.get_closest_point(robot_pose, valid_points)
        if new_point:
            if not self.frontier_point:
                self.frontier_point = robot_pose
            self.previous_frontier_point = copy.deepcopy(self.frontier_point)
            self.frontier_point = new_point
            self.total_exploration_time = data.exploration_time
            self.move_attempt = 0
            self.save_data([{'time': rospy.Time.now().secs, 'pose': robot_pose, 'frontier': new_point}],
                           'plots/frontiers_{}.pickle'.format(self.robot_id))

            if not self.is_initial_rendezvous_sharing:
                self.move_to_stop()
            else:
                self.is_initial_rendezvous_sharing = False
            self.move_robot_to_goal(self.frontier_point, direction)
            rospy.logerr("Robot {} going to frontier: {}".format(self.robot_id, self.frontier_point))
        else:
            rospy.logerr("Robot {} No frontier to go to...".format(self.robot_id))

    def wait_for_map_update(self):
        # r = rospy.Rate(0.1)
        while not self.is_time_to_moveon():
            # r.sleep()
            continue

    def publish_rendezvous_points(self, rendezvous_poses, receivers, direction=1, total_exploration_time=0):
        x = [float(v[0]) for v in rendezvous_poses]
        y = [float(v[1]) for v in rendezvous_poses]
        rv_points = RendezvousPoints()
        rv_points.x = x
        rv_points.y = y
        rv_points.header.frame_id = '{}'.format(self.robot_id)
        rv_points.exploration_time = total_exploration_time
        rv_points.direction = direction
        for id in receivers:
            self.rendezvous_publishers[id].publish(rv_points)
            sleep(1)  # wait for a second in order to order the RRs

    def robots_karto_out_callback(self, data):
        if data.robot_id - 1 == self.robot_id:
            if self.is_initial_data_sharing:
                self.push_messages_to_receiver(self.candidate_robots)
                self.previous_pose = self.get_robot_pose()
                self.is_initial_data_sharing = False
            else:
                for rid in self.candidate_robots:
                    self.add_to_file(rid, [data])

    def map_callback(self, data):
        self.last_map_update_time = rospy.Time.now().secs
        if not self.map_update_is_active:
            pose = self.get_robot_pose()
            self.map_update_is_active = True
            self.graph_processor.update_occupacygrid(data, pose)
            self.map_update_is_active = False
            if self.robot_id == 0:
                rospy.logerr("Robot {}: Map update NOT active".format(self.robot_id))

    def save_map_data(self, data):
        resolution = data.info.resolution
        width = data.info.width
        height = data.info.height
        origin_pos = data.info.origin.position
        grid_values = data.data
        origin_x = origin_pos.x
        origin_y = origin_pos.y
        row = {'resolution': resolution, 'width': width, 'height': height, 'values': grid_values, 'origin_x': origin_x,
               'origin_y': origin_y, 'known': self.graph_processor.unknowns, 'free': self.graph_processor.free_points,
               'obstacles': self.graph_processor.obstacles}
        self.save_data([row], 'map_message{}.pickle'.format(self.robot_id))

    def push_messages_to_receiver(self, receiver_ids, is_alert=0):
        for receiver_id in receiver_ids:
            message_data = self.load_data_for_id(receiver_id)
            buffered_data = BufferedData()
            buffered_data.header.frame_id = '{}'.format(self.robot_id)
            buffered_data.header.stamp.secs = rospy.Time.now().secs
            buffered_data.secs = []
            buffered_data.data = message_data
            buffered_data.alert_flag = is_alert
            self.publisher_map[receiver_id].publish(buffered_data)
            self.delete_data_for_id(receiver_id)

    def direction_has_changed(self):
        while self.others_active:
            continue
        self.others_active = True
        rospy.logerr("Robot {}: Computing direction".format(self.robot_id))
        self.check_data_sharing_status()
        self.others_active = False

    def check_data_sharing_status(self):
        robot_pose = self.get_robot_pose()
        P, intersections = self.graph_processor.compute_intersections((robot_pose[1], robot_pose[0]))
        time_stamp = rospy.Time.now().secs
        if P and intersections:
            if self.close_devices:
                self.is_sender = True
                for rid in self.close_devices:
                    rospy.logerr("Robot {} sending data to Robot {}".format(self.robot_id, rid))
                    self.push_messages_to_receiver([str(rid)])
                self.save_data([{'time': time_stamp, 'pose': robot_pose, 'intersections': intersections}],
                               'plots/interconnections_{}.pickle'.format(self.robot_id))

    def move_robot_to_goal(self, goal, direction=1):
        id_val = "robot_{}_{}_{}".format(self.robot_id, self.goal_count, direction)
        move = MoveToPosition2DActionGoal()
        move.header.frame_id = '/robot_{}/map'.format(self.robot_id)
        goal_id = GoalID()
        goal_id.id = id_val
        move.goal_id = goal_id
        move.goal.target_pose.x = goal[0]
        move.goal.target_pose.y = goal[1]
        self.moveTo_pub.publish(move)
        self.goal_count += 1
        chosen_point = ChosenPoint()
        chosen_point.header.frame_id = '{}'.format(self.robot_id)
        chosen_point.x = goal[0]
        chosen_point.y = goal[1]
        chosen_point.direction = direction
        self.chose_point_pub.publish(chosen_point)
        self.robot_state = PASSIVE_STATE

    def rotate_robot(self):
        deg_360 = math.radians(360)
        if self.robot_pose:
            while self.robot_pose[2] < deg_360:
                self.move_robot((0, 1))

    def move_status_callback(self, data):
        id_0 = "robot_{}_{}_{}".format(self.robot_id, self.goal_count - 1, TO_FRONTIER)
        if data.status_list:
            goal_status = data.status_list[0]
            if goal_status.goal_id.id:
                if goal_status.goal_id.id == id_0:
                    if goal_status.status == ACTIVE:
                        rospy.loginfo("Robot type, {} Heading to Frontier point...".format(self.robot_type))

    def signal_strength_callback(self, data):
        signals = data.signals
        time_stamp = data.header.stamp.secs
        robots = []
        devices = []
        for rs in signals:
            robots.append([rs.robot_id, rs.rssi])
            devices.append(rs.robot_id)
        self.signal_strength[time_stamp] = robots
        self.save_data([{'time': time_stamp, 'devices': robots}],
                       'plots/signal_strengths_{}.pickle'.format(self.robot_id))
        self.close_devices = devices

    def move_result_callback(self, data):
        id_0 = "robot_{}_{}_{}".format(self.robot_id, self.goal_count - 1, TO_FRONTIER)
        if data.status:
            if data.status.status == ABORTED:
                rospy.logerr("Robot {} failed to move".format(self.robot_id))
                if data.status.goal_id.id == id_0:
                    if self.move_attempt < MAX_ATTEMPTS:
                        rospy.logerr("Robot {}: Navigation failed. Trying again..".format(self.robot_id))
                        self.rotate_robot()
                        self.move_robot_to_goal(self.frontier_point, TO_FRONTIER)
                        self.move_attempt += 1
                    else:
                        result = self.start_exploration()
                        if result:
                            self.exploration_start_time = rospy.Time.now()
                            self.is_exploring = True
                            self.robot_state = ACTIVE_STATE

                        # rospy.logerr("Robot {}: Navigation failed!".format(self.robot_id))

            elif data.status.status == SUCCEEDED:
                if data.status.goal_id.id == id_0:
                    if self.robot_type == RR_TYPE:
                        rospy.logerr("Robot {} arrived at goal..".format(self.robot_id))
                        result = self.start_exploration()
                        if result:
                            self.exploration_start_time = rospy.Time.now()
                            self.is_exploring = True
                            self.robot_state = ACTIVE_STATE

    def process_data(self, sender_id, buff_data):
        data_vals = buff_data.data
        self.last_map_update_time = rospy.Time.now().secs
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
        if sender_id in self.candidate_robots:
            self.process_data(sender_id, buff_data)
        rospy.logerr("Robot {} received data from {}".format(self.robot_id, sender_id))
        self.wait_for_map_update()
        if self.initial_receipt:
            self.initial_data_count += 1
            if self.initial_data_count == len(self.candidate_robots):
                self.initial_receipt = False
                if self.i_have_least_id():
                    rospy.logerr("Robot {}: Computing frontier points...".format(self.robot_id))
                    robot_pose = self.get_robot_pose()
                    while self.map_update_is_active:
                        continue
                    self.others_active = True
                    rospy.logerr("Robot {}: Computing frontier points Active".format(self.robot_id))
                    frontier_points = self.graph_processor.get_frontiers((robot_pose[1], robot_pose[0]),
                                                                         len(self.candidate_robots) + 1)
                    rospy.logerr("Robot {}: Computing frontier points NOT Active".format(self.robot_id))
                    self.others_active = False
                    self.map_update_count = 0
                    if frontier_points:
                        self.send_frontier_points(robot_pose, frontier_points)
                        rospy.logerr("Robot {}: Moving to new frontier".format(self.robot_id))
                    else:
                        rospy.logerr("Robot {}: No valid frontier points".format(self.robot_id))
                else:
                    rospy.logerr("Robot {}: Waiting for frontier points...".format(self.robot_id))
        else:
            if not self.is_sender:
                self.push_messages_to_receiver([sender_id])
            else:
                self.received_devices += 1
                if self.received_devices == len(self.close_devices):
                    self.received_devices = 0
                    self.is_sender = False
                    robot_pose = self.get_robot_pose()
                    # if self.robot_id < int(sender_id):
                    rospy.logerr("Robot {} Recomputing frontier points".format(self.robot_id))
                    while self.map_update_is_active:
                        continue
                    self.others_active = True
                    frontier_points = self.graph_processor.get_frontiers((robot_pose[1], robot_pose[0]),
                                                                         len(self.candidate_robots) + 1)
                    self.others_active = False
                    self.map_update_count = 0
                    if frontier_points:
                        self.send_frontier_points(robot_pose, frontier_points)

    def send_frontier_points(self, robot_pose, frontier_points):
        self.publish_rendezvous_points(frontier_points, self.candidate_robots, direction=TO_FRONTIER)
        optimal_points = self.get_available_points(frontier_points)
        new_point = self.get_closest_point(robot_pose, optimal_points)
        if new_point:
            if not self.frontier_point:
                self.frontier_point = robot_pose
                self.previous_frontier_point = copy.deepcopy(self.frontier_point)
                self.frontier_point = new_point
                self.move_attempt = 0
                self.move_robot_to_goal(new_point, TO_FRONTIER)
                self.save_data([{'time': rospy.Time.now().secs, 'pose': robot_pose, 'frontier': new_point}],
                               'plots/frontiers_{}.pickle'.format(self.robot_id))
            else:
                rospy.logerr("Robot {}: No frontier points to send...".format(self.robot_id))

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
            distance[p] = self.graph_processor.D(robot_pose, p)
        if distance:
            closest_point = min(distance, key=distance.get)
        return closest_point

    def i_have_least_id(self):
        id = int(self.robot_id)
        least = min([int(i) for i in self.candidate_robots])
        return id < least

    def navigation_plan_callback(self, data):
        if self.waiting_for_plan:
            self.navigation_plan = data.cells

    def start_exploration_callback(self, data):
        self.start_now = data

    def get_total_navigation_time(self, origin, plan):
        total_time = 60  # 1 min by default
        if plan:
            ordered_points = self.graph_processor.get_distances(origin, plan)
            if ordered_points:
                farthest_point = ordered_points[-1]
                distance = math.sqrt((origin[0] - farthest_point[0]) ** 2 + (origin[1] - farthest_point[1]) ** 2)
                total_time = 2 * distance / 1.0  # distance/speed
        return total_time

    def is_time_to_moveon(self):
        diff = rospy.Time.now().secs - self.last_map_update_time
        return diff > 10

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
                robot_pose = (math.floor(robot_loc_val[0]), math.floor(robot_loc_val[1]))
                sleep(1)
            except:
                # rospy.logerr("Robot {}: Can't fetch robot pose from tf".format(self.robot_id))
                pass

        return robot_pose

    def exploration_callback(self, data):
        if data.status_list:
            goal_status = data.status_list[0]
            gid = goal_status.goal_id.id
            if gid and self.exploration_id != gid:
                self.exploration_id = gid
                status = goal_status.status
                if status == ABORTED:
                    rospy.logerr("Robot {}: Robot stopped share data..".format(self.robot_id))

    def cancel_exploration(self):
        if self.exploration_id:
            rospy.logerr("Robot {} Cancelling exploration...".format(self.robot_id))
            goal_id = GoalID()
            goal_id.id = self.exploration_id
            self.cancel_explore_pub.publish(goal_id)
        else:
            rospy.logerr("Exploration ID not set...")

    def add_map_to_file(self, rid, data):
        self.lock.acquire()
        if int(rid) == 1:
            filename = "map_messages1.pickle"
            saved_data = {}
            if not exists(filename):
                os.remove(filename)
                f = open(filename, "wb+")
                f.close()
            with open(filename, 'wb') as fp:
                pickle.dump(data[0], fp, protocol=pickle.HIGHEST_PROTOCOL)
            rospy.logerr("Robot {} Saved map message..".format(self.robot_id))
        sleep(2)
        self.lock.release()
        return True

    def add_to_file(self, rid, data):
        self.lock.acquire()
        if rid in self.karto_messages:
            self.karto_messages[rid] += data
        else:
            self.karto_messages[rid] = data
        self.lock.release()
        return True

    def load_data_for_id(self, rid):
        message_data = []
        if rid in self.karto_messages:
            message_data = self.karto_messages[rid]

        return message_data

    def load_map_data(self):
        filename = "map_messages1.pickle"
        data_dict = {}
        if exists(filename) and os.path.getsize(filename) > 0:
            with open(filename, 'rb') as fp:
                try:
                    data_dict = pickle.load(fp)
                except Exception as e:
                    rospy.logerr("error: {}".format(e))
        return data_dict

    def load_data(self):
        return self.karto_messages

    def update_file(self, data):
        with open(self.data_file_name, 'wb+') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    def delete_data_for_id(self, rid):
        self.lock.acquire()
        if rid in self.karto_messages:
            del self.karto_messages[rid]
        self.lock.release()
        return True

    def pose_callback(self, msg):
        pose = (msg.x, msg.y, msg.theta)
        self.robot_pose = pose

    def move_robot(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.angular.z = vel[1]
        self.pose_publisher.publish(vel_msg)

    def clear_data(self):
        if exists(self.data_file_name):
            os.remove(self.data_file_name)
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
                    rospy.logerr("error: {}".format(e))
        return data_dict


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
