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
from gvgexploration.msg import *
from gvgexploration.srv import *
import numpy as np
from time import sleep
from threading import Lock
import tf
import copy
import project_utils as pu
from actionlib import SimpleActionClient
from std_msgs.msg import String
from Queue import Queue

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
        self.received_devices = []
        self.current_devices = []
        self.is_initial_sharing = True
        self.is_sender = False
        self.previous_pose = None
        self.is_initial_data_sharing = True
        self.initial_receipt = True
        self.is_initial_move = True
        self.last_map_update_time = rospy.Time.now().secs
        self.frontier_points = []
        self.coverage = None
        self.goal_handle = None
        self.auction_feedback = {}
        self.all_feedbacks = {}

        # communication synchronizatin variables
        self.comm_session_time = 0
        self.auction_session_time = 0
        self.time_after_bidding = 0
        self.auction_waiting_time = 0
        self.waiting_for_auction_feedback = False
        self.waiting_for_frontier_point = False
        self.waiting_for_auction = False
        self.waiting_for_response = False
        self.session_id = None
        # communinication session variables end here

        self.exploration_time = rospy.Time.now().to_sec()
        self.candidate_robots = self.frontier_robots + self.base_stations
        self.frontier_data = []
        self.interconnection_data = []
        self.rate = rospy.Rate(0.1)
        self.run = rospy.get_param("~run")
        self.termination_metric = rospy.get_param("~termination_metric")
        self.max_exploration_time = rospy.get_param("~max_exploration_time")
        self.max_coverage = rospy.get_param("~max_coverage")
        self.max_common_coverage = rospy.get_param("~max_common_coverage")
        self.max_wait_time = rospy.get_param("~max_wait_time")
        self.environment = rospy.get_param("~environment")
        self.robot_count = rospy.get_param("~robot_count")
        for rid in self.candidate_robots:
            pub = rospy.Publisher("/roscbt/robot_{}/received_data".format(rid), BufferedData, queue_size=100)
            pub1 = rospy.Publisher("/roscbt/robot_{}/auction_points".format(rid), Auction, queue_size=10)
            pub2 = rospy.Publisher("/roscbt/robot_{}/allocated_point".format(rid), Frontier, queue_size=10)
            pub3 = rospy.Publisher("/roscbt/robot_{}/auction_feedback".format(rid), Auction, queue_size=10)
            self.allocation_pub[rid] = pub2
            self.publisher_map[rid] = pub
            self.auction_publishers[rid] = pub1
            self.auction_feedback_pub[rid] = pub3

        self.karto_pub = rospy.Publisher("/robot_{}/karto_in".format(self.robot_id), LocalizedScan, queue_size=100)
        rospy.Subscriber('/robot_{}/received_data'.format(self.robot_id), BufferedData, self.buffered_data_callback,
                         queue_size=10)
        rospy.Subscriber("/robot_{}/signal_strength".format(self.robot_id), SignalStrength,
                         self.signal_strength_callback)
        rospy.Subscriber('/karto_out'.format(self.robot_id), LocalizedScan, self.robots_karto_out_callback,
                         queue_size=10)
        rospy.Subscriber("/robot_{}/auction_points".format(self.robot_id), Auction, self.auction_points_callback)
        self.fetch_frontier_points = rospy.ServiceProxy('/robot_{}/frontier_points'.format(self.robot_id),
                                                        FrontierPoint)
        self.check_intersections = rospy.ServiceProxy('/robot_{}/check_intersections'.format(self.robot_id),
                                                      Intersections)
        rospy.Subscriber('/coverage'.format(self.robot_id), Coverage, self.coverage_callback)
        rospy.Subscriber('/robot_{}/allocated_point'.format(self.robot_id), Frontier, self.allocated_point_callback)
        rospy.Subscriber('/robot_{}/auction_feedback'.format(self.robot_id), Auction, self.auction_feedback_callback)
        rospy.Subscriber('/robot_{}/map'.format(self.robot_id), OccupancyGrid, self.map_update_callback)
        rospy.Subscriber('/robot_{}/gvg_explore/feedback'.format(self.robot_id), GvgExploreActionFeedback,
                         self.explore_feedback_callback)
        self.shutdown_pub = rospy.Publisher("/shutdown".format(self.robot_id), String, queue_size=10)
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)
        self.is_shutdown_caller = False

        # robot identification sensor
        self.robot_range_pub = rospy.Publisher("/robot_{}/robot_ranges".format(self.robot_id), RobotRange, queue_size=5)

        self.trans_matrices = {}
        self.inverse_trans_matrices = {}
        self.prev_intersects = []

        # ======= pose transformations====================
        self.listener = tf.TransformListener()
        self.exploration = SimpleActionClient("/robot_{}/gvg_explore".format(self.robot_id), GvgExploreAction)
        self.exploration.wait_for_server()
        rospy.loginfo("Robot {} Initialized successfully!!".format(self.robot_id))
        rospy.on_shutdown(self.save_all_data)
        self.first_message_sent = False
        self.sent_messages = []
        self.received_messages = []


    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():

            if self.is_exploring and not self.session_id:
                rospy.logerr('Robot {}: Is exploring: {}, Session ID: {}'.format(self.robot_id, self.is_exploring,self.session_id))
                self.check_data_sharing_status()
                time_to_shutdown = self.evaluate_exploration()
                if time_to_shutdown:
                    self.cancel_exploration()
                    rospy.signal_shutdown('Exploration complete!')
            # else:
            #     time_to_go = self.evaluate_waiting()
            #     rospy.logerr('Robot {}: Is exploring: {}, Session ID: {}'.format(self.robot_id, self.is_exploring,
            #                                                                      self.session_id))
            #     if time_to_go:
            #         self.session_id = None  # you've  dropped the former commitment
            #         self.resume_exploration()  # continue with your exploration
            r.sleep()

    def evaluate_waiting(self):
        time_to_go = False
        if self.waiting_for_response:  # after sending data
            if (rospy.Time.now().to_sec() - self.comm_session_time) > self.max_wait_time:
                # self.request_and_share_frontiers()  # just work with whoever has replied
                self.waiting_for_response = False
                self.comm_session_time = rospy.Time.now().to_sec()
                time_to_go = True

        if self.waiting_for_frontier_point:  # after advertising an auction
            if (rospy.Time.now().to_sec() - self.time_after_bidding) > self.max_wait_time:
                self.waiting_for_frontier_point = False
                self.waiting_for_frontier_point = rospy.Time.now().to_sec()
                time_to_go = True

        if self.waiting_for_auction_feedback:  # after bidding in an auction
            if (rospy.Time.now().to_sec() - self.auction_session_time) > self.max_wait_time:
                self.waiting_for_auction_feedback = False
                self.auction_session_time = rospy.Time.now().to_sec()
                time_to_go = True

        if self.waiting_for_auction:  # after sending back data
            if (rospy.Time.now().to_sec() - self.auction_waiting_time) > self.max_wait_time:
                self.waiting_for_auction = False
                self.auction_waiting_time = rospy.Time.now().to_sec()
                time_to_go = True
        return time_to_go

    def resume_exploration(self):
        pose = self.get_robot_pose()
        p = Pose()
        p.position.x = pose[pu.INDEX_FOR_X]
        p.position.y = pose[pu.INDEX_FOR_Y]
        self.frontier_point = p
        self.start_exploration_action(p)

    def evaluate_exploration(self):
        its_time = False
        if self.coverage:
            if self.termination_metric == pu.MAXIMUM_EXPLORATION_TIME:
                time = rospy.Time.now().to_sec() - self.exploration_time
                its_time = time >= self.max_exploration_time * 60
            elif self.termination_metric == pu.TOTAL_COVERAGE:
                its_time = self.coverage.coverage >= self.max_coverage
            elif self.termination_metric == pu.FULL_COVERAGE:
                its_time = self.coverage.coverage >= self.coverage.expected_coverage
            elif self.termination_metric == pu.COMMON_COVERAGE:
                its_time = self.coverage.common_coverage >= self.max_common_coverage

        return its_time

    def check_data_sharing_status(self):
        robot_pose = self.get_robot_pose()
        p = Pose()
        p.position.x = robot_pose[pu.INDEX_FOR_X]
        p.position.y = robot_pose[pu.INDEX_FOR_Y]
        response = self.check_intersections(IntersectionsRequest(pose=p))
        time_stamp = rospy.Time.now().to_sec()
        if response.result:
            if self.close_devices and not self.session_id:  # devices available and you're not in session
                self.current_devices = copy.deepcopy(self.close_devices)
                session_id = '{}_{}'.format(self.robot_id, rospy.Time.now().to_sec())
                self.session_id = session_id
                self.is_sender = True
                self.is_exploring = False
                self.comm_session_time = rospy.Time.now().to_sec()
                self.waiting_for_response = True
                rospy.logerr("Robot {}: Sending data to available devices...{}".format(self.robot_id,self.current_devices))
                # # ===== ends here ====================
                self.push_messages_to_receiver(self.current_devices, session_id, initiator=1)
                self.interconnection_data.append( {'time': time_stamp, 'pose': robot_pose, 'intersection_range': response.result})

    def explore_feedback_callback(self, data):
        self.is_exploring = True
        self.session_id = None

    def map_update_callback(self, data):
        self.last_map_update_time = rospy.Time.now().to_sec()

    def wait_for_updates(self):
        sleep_time = rospy.Time.now().to_sec() - self.last_map_update_time
        while sleep_time < 5:
            rospy.sleep(sleep_time)
            sleep_time = rospy.Time.now().to_sec() - self.last_map_update_time

    def coverage_callback(self, data):
        self.coverage = data

    def allocated_point_callback(self, data):
        rospy.logerr("Robot {}: received allocated points".format(self.robot_id))
        if not self.is_sender and data.session_id.data == self.session_id:
            # reset waiting for frontier point flags
            self.waiting_for_frontier_point = False
            self.time_after_bidding = rospy.Time.now().to_sec()
            # ============ End here =================
            self.frontier_point = data.pose
            robot_pose = self.get_robot_pose()
            new_point = (data.pose.position.x, data.pose.position.y, 0)
            self.frontier_data.append(
                {'time': rospy.Time.now().to_sec(), 'distance_to_frontier': pu.D(robot_pose, new_point)})
            self.start_exploration_action(self.frontier_point)

    def auction_feedback_callback(self, data):
        rospy.logerr("Robot {}: received auction feedback".format(self.robot_id))
        if self.is_sender and data.session_id.data == self.session_id:
            self.all_feedbacks[data.msg_header.header.frame_id] = data
            rid = data.msg_header.header.frame_id
            m = len(data.distances)
            min_dist = max(data.distances)
            min_pose = None
            for i in range(m):
                if min_dist >= data.distances[i]:
                    min_pose = data.poses[i]
                    min_dist = data.distances[i]
            self.auction_feedback[rid] = (min_dist, min_pose)
            if len(self.all_feedbacks) >= len(self.close_devices):
                all_robots = list(self.auction_feedback)
                taken_poses = []
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
                    frontier.msg_header.header.frame_id = '{}'.format(self.robot_id)
                    frontier.msg_header.header.stamp = rospy.Time.now()
                    frontier.msg_header.sender_id = str(self.robot_id)
                    frontier.msg_header.receiver_id = str(i)
                    frontier.msg_header.topic = 'allocated_point'
                    frontier.pose = rpose
                    frontier.session_id = data.session_id
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
                                remaining_poses[rob_distances[k]] = rob_poses[k]
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
                if not self.frontier_point:
                    rospy.logerr("Robot {}: No frontier point left! ".format(self.robot_id))
                    pose = Pose()
                    pose.position.x = robot_pose[pu.INDEX_FOR_X]
                    pose.position.y = robot_pose[pu.INDEX_FOR_Y]
                    self.frontier_point = pose

                # reset waiting for bids flats and stop being a sender
                self.waiting_for_auction_feedback = False
                self.auction_session_time = rospy.Time.now().to_sec()
                self.is_sender = False
                # =============Ends here ==============
                self.start_exploration_action(self.frontier_point)
                self.all_feedbacks.clear()
                self.auction_feedback.clear()
                self.received_devices = []

    def auction_points_callback(self, data):
        rospy.logerr("Robot {}: received auction points".format(self.robot_id))
        if not self.is_sender:  # only participate if you're not a sender
            if self.session_id and data.session_id.data == self.session_id:
                sender_id = data.msg_header.header.frame_id
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
                auction.msg_header.header.frame_id = '{}'.format(self.robot_id)
                auction.msg_header.header.stamp = rospy.Time.now()
                auction.msg_header.sender_id = str(self.robot_id)
                auction.msg_header.receiver_id = str(sender_id)
                auction.msg_header.topic = 'auction_feedback'
                auction.poses = received_points
                auction.distances = distances
                auction.session_id = data.session_id
                self.auction_feedback_pub[sender_id].publish(auction)

                # start waiting for location after bidding
                self.time_after_bidding = rospy.Time.now().to_sec()
                self.waiting_for_frontier_point = True

                # reset waiting for auction flags
                self.waiting_for_auction = False
                self.auction_waiting_time = rospy.Time.now().to_sec()

    def publish_auction_points(self, rendezvous_poses, receivers, direction=1, total_exploration_time=0):
        auction = Auction()
        auction.msg_header.header.frame_id = '{}'.format(self.robot_id)
        auction.msg_header.sender_id = str(self.robot_id)
        auction.msg_header.topic = 'auction_points'
        auction.msg_header.header.stamp = rospy.Time.now()
        auction.session_id.data = self.session_id
        auction.poses = []
        self.auction_feedback.clear()
        self.all_feedbacks.clear()
        for v in rendezvous_poses:
            pose = Pose()
            pose.position.x = v[pu.INDEX_FOR_X]
            pose.position.y = v[pu.INDEX_FOR_Y]
            auction.poses.append(pose)
        for id in receivers:
            auction.msg_header.receiver_id = str(id)
            self.auction_publishers[id].publish(auction)
        rospy.logerr('Robot {} published auction..{}'.format(self.robot_id, rendezvous_poses))
        # wait for bids from recipients
        self.auction_session_time = rospy.Time.now().to_sec()
        self.waiting_for_auction_feedback = True

    def robots_karto_out_callback(self, data):
        if data.robot_id - 1 == self.robot_id:
            for rid in self.candidate_robots:
                self.add_to_file(rid, [data])
            if self.is_initial_data_sharing:
                self.is_initial_data_sharing = False
                self.push_messages_to_receiver(self.candidate_robots, None, initiator=1)

    def push_messages_to_receiver(self, receiver_ids, session_id, is_alert=0, initiator=0):
        for receiver_id in receiver_ids:
            message_data = self.load_data_for_id(receiver_id)
            buffered_data = BufferedData()
            if session_id:
                buffered_data.session_id.data = session_id
            else:
                buffered_data.session_id.data = ''
            buffered_data.msg_header.header.frame_id = '{}'.format(self.robot_id)
            buffered_data.msg_header.sender_id = str(self.robot_id)
            buffered_data.msg_header.receiver_id = str(receiver_id)
            buffered_data.msg_header.topic = 'received_data'
            buffered_data.msg_header.header.stamp = rospy.Time.now()
            buffered_data.secs = []
            buffered_data.data = message_data
            buffered_data.alert_flag = is_alert
            self.publisher_map[str(receiver_id)].publish(buffered_data)
            self.sent_messages.append(
                {'time': rospy.Time.now().to_sec(), 'robot_id': self.robot_id, 'is_initiator': initiator,
                 'session_id': self.session_id})
            self.delete_data_for_id(receiver_id)

    def signal_strength_callback(self, data):
        signals = data.signals
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
        # for rid in self.candidate_robots:
        #     if rid != sender_id and rid not in self.close_devices:# if a given robot is close, the we assume it received
        #         self.add_to_file(rid, data_vals)

    def buffered_data_callback(self, buff_data):
        sender_id = buff_data.msg_header.header.frame_id
        now = rospy.Time.now().to_sec()
        sent_time = buff_data.msg_header.header.stamp.to_sec()
        time_diff = now - sent_time
        self.received_messages.append(
            {'time': now, 'robot_id': self.robot_id, 'session_id': buff_data.session_id.data, 'time_diff': time_diff,
             'topic': buff_data.msg_header.topic})
        self.process_data(sender_id, buff_data)
        # ============ used during initialization ============
        if self.initial_receipt and not buff_data.session_id.data:
            # self.wait_for_updates()
            self.initial_data_count += 1
            if self.initial_data_count == len(self.candidate_robots):
                self.initial_receipt = False
                if self.i_have_least_id():
                    self.is_sender = True
                    self.session_id = '{}_{}'.format(self.robot_id, rospy.Time.now().to_sec())
                    self.comm_session_time = rospy.Time.now().to_sec()
                    self.waiting_for_response = True
                    self.current_devices = copy.deepcopy(self.close_devices)
                    self.push_messages_to_receiver(self.candidate_robots, self.session_id, initiator=1)
                    rospy.logerr('Robot {} initiating coordination..'.format(self.robot_id))

                else:
                    rospy.logerr("Robot {}: Waiting for frontier points...".format(self.robot_id))

        # ===============the block below is used during exploration =====================
        else:
            if buff_data.session_id.data:  # only respond to data with session id
                rospy.logerr('Robot {}: Received session data from {}'.format(self.robot_id, sender_id))
                if not self.initial_receipt:
                    self.initial_receipt = False
                if not self.session_id:  # send back data and stop to wait for further communication
                    self.session_id = buff_data.session_id.data
                    rospy.logerr('Robot {}: Received new session id {}'.format(self.robot_id, self.session_id))
                    self.push_messages_to_receiver([sender_id], self.session_id, initiator=0)
                    self.is_exploring = False
                    self.cancel_exploration()

                    # set auction waiting flags
                    self.waiting_for_auction = True
                    self.auction_waiting_time = rospy.Time.now().to_sec()

                else:  # So you have a session id
                    if buff_data.session_id.data == self.session_id:
                        if self.is_sender:  # do this if you're a sender
                            self.received_devices.append(sender_id)
                            rospy.logerr("Robot {}: Received devices {}, Available devices: {}".format(self.robot_id,
                                                                                                       self.received_devices,
                                                                                                       self.current_devices))
                            if len(self.received_devices) == len(self.current_devices):
                                rospy.logerr(
                                    'Robot {}:Received all expected responses...computing new locations now'.format(
                                        self.robot_id))
                                self.request_and_share_frontiers()
                        else:
                            rospy.logerr('Robot {}: I am not the sender..ignoring session data'.format(self.robot_id))

    def request_and_share_frontiers(self):
        frontier_point_response = self.fetch_frontier_points(FrontierPointRequest(count=len(self.received_devices) + 1))
        frontier_points = self.parse_frontier_response(frontier_point_response)
        if frontier_points:
            self.frontier_points = frontier_points
            self.publish_auction_points(frontier_points, self.close_devices, direction=TO_FRONTIER)

            # set auction waiting flags
            self.auction_session_time = rospy.Time.now().to_sec()
            self.waiting_for_auction_feedback = True
        else:
            self.received_devices = []
            self.is_sender = False  # reset this: you're not a sender anymore
            self.start_exploration_action()  # say nothing, just go your way

        # reset the waiting flags, say nothing and move
        self.waiting_for_response = False
        self.comm_session_time = rospy.Time.now().to_sec()

    def start_exploration_action(self, pose):
        goal = GvgExploreGoal(pose=pose)
        self.exploration.wait_for_server()
        self.goal_handle = self.exploration.send_goal(goal)
        self.exploration_time = rospy.Time.now().to_sec()
        self.exploration.wait_for_result()
        self.exploration.get_result()

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
        self.is_exploring = False
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

    def get_elevation(self, quaternion):
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        return yaw

    def save_all_data(self):
        pu.save_data(self.interconnection_data,
                     'gvg/interconnections_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                                         self.termination_metric, self.robot_id))
        pu.save_data(self.frontier_data,
                     'gvg/frontiers_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                                  self.termination_metric, self.robot_id))

        pu.save_data(self.sent_messages,
                     'gvg/sent_messages_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                                      self.termination_metric, self.robot_id))
        pu.save_data(self.received_messages,
                     'gvg/received_messages_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                                          self.termination_metric, self.robot_id))
        msg = String()
        msg.data = '{}'.format(self.robot_id)
        self.is_shutdown_caller = True
        self.shutdown_pub.publish(msg)

    def shutdown_callback(self, data):
        if not self.is_shutdown_caller:
            rospy.signal_shutdown('Robot {}: Received Shutdown Exploration complete!'.format(self.robot_id))


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
