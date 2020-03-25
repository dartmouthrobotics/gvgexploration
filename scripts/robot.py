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
from threading import Thread
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
        self.frontier_ridge = None
        self.robot_state = ACTIVE_STATE
        self.publisher_map = {}
        self.allocation_pub = {}
        self.shared_data_srv_map = {}
        self.shared_point_srv_map = {}
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

        self.buff_data_srv = rospy.Service('/robot_{}/shared_data'.format(self.robot_id), SharedData,
                                           self.shared_data_handler)
        self.auction_points_srv = rospy.Service("/robot_{}/auction_points".format(self.robot_id), SharedPoint,
                                                self.shared_point_handler)
        rospy.Subscriber('/robot_{}/allocated_point'.format(self.robot_id), Frontier, self.allocated_point_callback)
        rospy.Subscriber('/robot_{}/received_data'.format(self.robot_id), BufferedData,
                         self.initial_data_callback)  # just for initial data *
        self.karto_pub = rospy.Publisher("/robot_{}/karto_in".format(self.robot_id), LocalizedScan, queue_size=10)
        self.signal_strength_srv = rospy.ServiceProxy("/signal_strength".format(self.robot_id), HotSpot)
        self.fetch_frontier_points = rospy.ServiceProxy('/robot_{}/frontier_points'.format(self.robot_id),
                                                        FrontierPoint)
        self.check_intersections = rospy.ServiceProxy('/robot_{}/check_intersections'.format(self.robot_id),
                                                      Intersections)
        rospy.Subscriber('/coverage'.format(self.robot_id), Coverage, self.coverage_callback)
        rospy.Subscriber('/robot_{}/map'.format(self.robot_id), OccupancyGrid, self.map_update_callback)
        rospy.Subscriber('/robot_{}/gvgexplore/feedback'.format(self.robot_id), GvgExploreActionFeedback,
                         self.explore_feedback_callback)

        for rid in self.candidate_robots:
            received_data_clt = rospy.ServiceProxy("/robot_{}/shared_data".format(rid), SharedData)
            action_points_clt = rospy.ServiceProxy("/robot_{}/auction_points".format(rid), SharedPoint)
            alloc_point_pub = rospy.Publisher("/roscbt/robot_{}/allocated_point".format(rid), Frontier, queue_size=10)
            pub = rospy.Publisher("/roscbt/robot_{}/received_data".format(rid), BufferedData, queue_size=10)
            self.publisher_map[rid] = pub
            self.allocation_pub[rid] = alloc_point_pub
            self.shared_data_srv_map[rid] = received_data_clt
            self.shared_point_srv_map[rid] = action_points_clt
        rospy.Subscriber('/karto_out', LocalizedScan, self.robots_karto_out_callback,
                         queue_size=10)
        self.shutdown_pub = rospy.Publisher("/shutdown".format(self.robot_id), String, queue_size=10)
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)
        self.is_shutdown_caller = False

        self.trans_matrices = {}
        self.inverse_trans_matrices = {}
        self.prev_intersects = []

        # ======= pose transformations====================
        self.listener = tf.TransformListener()
        self.exploration = SimpleActionClient("/robot_{}/gvgexplore".format(self.robot_id), GvgExploreAction)
        self.exploration.wait_for_server()
        rospy.loginfo("Robot {} Initialized successfully!!".format(self.robot_id))
        # rospy.on_shutdown(self.save_all_data)
        self.first_message_sent = False
        self.sent_messages = []
        self.received_messages = []

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            rospy.logerr('Robot {}: Is exploring: {}, Session ID: {}'.format(self.robot_id, self.is_exploring, self.session_id))
            if self.is_exploring:
                self.check_data_sharing_status()
            r.sleep()

    def resume_exploration(self):
        pose = self.get_robot_pose()
        p = Pose()
        p.position.x = pose[pu.INDEX_FOR_X]
        p.position.y = pose[pu.INDEX_FOR_Y]
        self.frontier_ridge = p
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
        rospy.logerr("Robot {}: Intersec response..{}".format(self.robot_id,response))
        time_stamp = rospy.Time.now().to_sec()
        if response.result:
            close_devices = self.get_close_devices()
            if close_devices and not self.session_id:  # devices available and you're not in session
                self.interconnection_data.append(
                    {'time': time_stamp, 'pose': robot_pose, 'intersection_range': response.result})
                self.handle_intersection(close_devices)

    def handle_intersection(self, current_devices):
        session_id = '{}_{}'.format(self.robot_id, rospy.Time.now().to_sec())
        self.session_id = session_id
        self.is_sender = True
        self.cancel_exploration()
        session_devices = []
        buff_data = {}
        for rid in current_devices:
            message_data = self.load_data_for_id(rid)
            buffered_data = self.create_buffered_data_msg(message_data, session_id, rid)
            response = self.shared_data_srv_map[rid](SharedDataRequest(req_data=buffered_data))
            rospy.logerr("Robot {}: received feedback from robot: {}".format(self.robot_id, response.in_session))
            if not response.in_session:
                session_devices.append(rid)
                received_data = response.res_data
                buff_data[rid] = received_data
        self.process_data(buff_data)
        frontier_point_response = self.fetch_frontier_points(FrontierPointRequest(count=len(current_devices) + 1))
        frontier_points = self.parse_frontier_response(frontier_point_response)
        taken_poses = []
        if frontier_points:
            if session_devices:
                auction = self.create_auction(frontier_points)
                auction_feedback = {}
                for rid in session_devices:
                    auction_response = self.shared_point_srv_map[rid](SharedPointRequest(req_data=auction))
                    if auction_response.auction_accepted:
                        rospy.logerr("Robot {}: received auction feedback".format(self.robot_id))
                        data = auction_response.res_data
                        self.all_feedbacks[rid] = data
                        m = len(data.distances)
                        min_dist = max(data.distances)
                        min_pose = None
                        for i in range(m):
                            if min_dist >= data.distances[i]:
                                min_pose = data.poses[i]
                                min_dist = data.distances[i]
                        auction_feedback[rid] = (min_dist, min_pose)
                taken_poses = self.compute_and_share_auction_points(auction_feedback, frontier_points)
        else:
            rospy.logerr("Robot {}: just send an empty message so that the robots move on".format(self.robot_id))
            for rid in current_devices:
                auction = self.create_auction(frontier_points)
                self.shared_point_srv_map[rid](SharedPointRequest(req_data=auction))

        self.frontier_ridge = self.compute_next_frontier(taken_poses, frontier_points)
        if self.frontier_ridge:
            self.start_exploration_action(self.frontier_ridge)
        else:
            rospy.logerr("Robot {}: No frontier point left".format(self.robot_id))
        self.is_sender = False
        self.all_feedbacks.clear()
        # =============Ends here ==============

    def compute_next_frontier(self, taken_poses, frontier_points):
        ridge = None
        robot_pose = self.get_robot_pose()
        self.frontier_ridge = None
        dist_dict = {}
        for point in frontier_points:
            dist_dict[pu.D(robot_pose, point)] = point
        while not self.frontier_ridge and dist_dict:
            min_dist = min(list(dist_dict))
            closest = dist_dict[min_dist]
            if closest not in taken_poses:
                ridge = frontier_points[closest]
                break
            del dist_dict[min_dist]
        return ridge

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

        if not self.is_sender and data.session_id == self.session_id:
            # reset waiting for frontier point flags
            self.waiting_for_frontier_point = False
            self.time_after_bidding = rospy.Time.now().to_sec()
            # ============ End here =================
            self.frontier_ridge = data.ridge
            new_point = [0.0] * 2
            new_point[pu.INDEX_FOR_X] = self.frontier_ridge.nodes[1].position.x
            new_point[pu.INDEX_FOR_Y] = self.frontier_ridge.nodes[1].position.y
            robot_pose = self.get_robot_pose()
            self.frontier_data.append(
                {'time': rospy.Time.now().to_sec(), 'distance_to_frontier': pu.D(robot_pose, new_point)})
            rospy.logerr("Robot {}: received allocated points".format(self.robot_id))
            self.start_exploration_action(self.frontier_ridge)

    def compute_and_share_auction_points(self, auction_feedback, frontier_points):
        all_robots = list(auction_feedback)
        taken_poses = []
        for i in all_robots:
            pose_i = (auction_feedback[i][1].position.x, auction_feedback[i][1].position.y)
            conflicts = []
            for j in all_robots:
                if i != j:
                    pose_j = (auction_feedback[j][1].position.x, self.auction_feedback[j][1].position.y)
                    if pose_i == pose_j:
                        conflicts.append((i, j))
            rpose = auction_feedback[i][1]
            frontier = self.create_frontier(i, frontier_points[(rpose.position.x, rpose.position.y)])
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
                auction_feedback[conflicting_robot_id] = (next_closest_dist, remaining_poses[next_closest_dist])
        return taken_poses

    def create_frontier(self, receiver, ridge):
        frontier = Frontier()
        frontier.msg_header.header.frame_id = '{}'.format(self.robot_id)
        frontier.msg_header.header.stamp = rospy.Time.now()
        frontier.msg_header.sender_id = str(self.robot_id)
        frontier.msg_header.receiver_id = str(receiver)
        frontier.msg_header.topic = 'allocated_point'
        frontier.ridge = ridge  #
        frontier.session_id = self.session_id
        return frontier

    def create_auction(self, rendezvous_poses, distances=[]):
        auction = Auction()
        auction.msg_header.header.frame_id = '{}'.format(self.robot_id)
        auction.msg_header.sender_id = str(self.robot_id)
        auction.msg_header.topic = 'auction_points'
        auction.msg_header.header.stamp = rospy.Time.now()
        if not self.session_id:
            auction.session_id = ''
        auction.session_id = self.session_id
        auction.distances = distances
        auction.poses = []
        self.auction_feedback.clear()
        self.all_feedbacks.clear()
        for k, v in rendezvous_poses.items():
            pose = Pose()
            pose.position.x = k[pu.INDEX_FOR_X]
            pose.position.y = k[pu.INDEX_FOR_Y]
            auction.poses.append(pose)
        return auction

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
            buffered_data = self.create_buffered_data_msg(message_data, session_id, receiver_id)
            self.publisher_map[str(receiver_id)].publish(buffered_data)
            self.delete_data_for_id(receiver_id)

    def create_buffered_data_msg(self, message_data, session_id, receiver_id):
        buffered_data = BufferedData()
        if session_id:
            buffered_data.session_id = session_id
        else:
            buffered_data.session_id = ''
        buffered_data.msg_header.header.frame_id = '{}'.format(self.robot_id)
        buffered_data.msg_header.sender_id = str(self.robot_id)
        buffered_data.msg_header.receiver_id = str(receiver_id)
        buffered_data.msg_header.topic = 'received_data'
        buffered_data.msg_header.header.stamp = rospy.Time.now()
        buffered_data.secs = []
        buffered_data.data = message_data
        return buffered_data

    def get_close_devices(self):
        ss_data = self.signal_strength_srv(HotSpotRequest(robot_id=str(self.robot_id)))
        data = ss_data.hot_spots
        signals = data.signals
        robots = []
        devices = []
        for rs in signals:
            robots.append([rs.robot_id, rs.rssi])
            devices.append(str(rs.robot_id))
        return set(devices)

    def process_data(self, buff_data):
        counter = 0
        for rid, rdata in buff_data.items():
            data_vals = rdata.data
            for scan in data_vals:
                self.karto_pub.publish(scan)
                counter += 1
        sleep(counter / 2.0)

    def shared_data_handler(self, data):
        buff_data = data.req_data
        received_data = {buff_data.msg_header.sender_id: buff_data}
        thread = Thread(target=self.process_data, args=(received_data,))
        thread.start()
        sender_id = buff_data.msg_header.sender_id
        session_id = buff_data.session_id
        message_data = self.load_data_for_id(sender_id)
        not_available = 0
        if self.session_id:
            not_available = 1
        else:
            self.cancel_exploration()
            self.session_id = session_id
        buff_data = self.create_buffered_data_msg(message_data, session_id, sender_id)
        return SharedDataResponse(in_session=not_available, res_data=buff_data)

    def shared_point_handler(self, auction_data):
        data = auction_data.req_data
        session_id = data.session_id
        if self.is_sender or not self.session_id or session_id != self.session_id:
            return SharedPointResponse(auction_accepted=0)
        sender_id = data.msg_header.header.frame_id
        poses = data.poses
        if not poses:
            rospy.logerr("Robot {}: No poses received. Proceeding to my next frontier".format(self.robot_id))
            self.start_exploration_action(self.frontier_ridge)
            return SharedPointResponse(auction_accepted=1, res_data=None)
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
        # start waiting for location after bidding
        self.time_after_bidding = rospy.Time.now().to_sec()
        self.waiting_for_frontier_point = True

        # reset waiting for auction flags
        self.waiting_for_auction = False
        self.auction_waiting_time = rospy.Time.now().to_sec()
        return SharedPointResponse(auction_accepted=1, res_data=auction)

    def initial_data_callback(self, buff_data):
        sender_id = buff_data.msg_header.header.frame_id
        self.process_data({sender_id: buff_data})
        # ============ used during initialization ============
        if self.initial_receipt and not buff_data.session_id:
            # self.wait_for_updates()
            self.initial_data_count += 1
            if self.initial_data_count == len(self.candidate_robots):
                self.initial_receipt = False
                if self.i_have_least_id():
                    self.is_sender = True
                    self.session_id = '{}_{}'.format(self.robot_id, rospy.Time.now().to_sec())
                    self.comm_session_time = rospy.Time.now().to_sec()
                    self.waiting_for_response = True
                    close_devices = self.get_close_devices()
                    if close_devices:
                        self.handle_intersection(close_devices)
                else:
                    rospy.logerr("Robot {}: Waiting for frontier points...".format(self.robot_id))

    def start_exploration_action(self, frontier_ridge):
        goal = GvgExploreGoal(ridge=frontier_ridge)
        # self.exploration.wait_for_server()
        self.goal_handle = self.exploration.send_goal(goal)
        self.exploration.wait_for_result()
        self.exploration.get_result()

    def parse_frontier_response(self, data):
        frontier_points = {}
        received_ridges = data.ridges
        for r in received_ridges:
            p = r.nodes[1]
            frontier_points[(p.position.x, p.position.y)] = r
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
        if self.is_exploring:
            self.exploration.cancel_all_goals()
            self.is_exploring = False
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

        # pu.save_data(self.sent_messages,
        #              'gvg/sent_messages_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
        #                                                               self.termination_metric, self.robot_id))
        # pu.save_data(self.received_messages,
        #              'gvg/received_messages_{}_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
        #                                                                   self.termination_metric, self.robot_id))
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
