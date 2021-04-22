#!/usr/bin/python

import time
import sys
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
from std_msgs.msg import String
from scipy.optimize import linear_sum_assignment

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
        self.change_gate_alert = {}
        self.shared_data_srv_map = {}
        self.shared_point_srv_map = {}
        self.auction_feedback_pub = {}
        self.initial_data = set()
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
        self.last_map_update_time = rospy.Time.now().to_sec()
        self.frontier_points = []
        self.coverage = None
        self.goal_handle = None
        self.all_feedbacks = {}
        # communication synchronizatin variables
        self.comm_session_time = 0
        self.auction_session_time = 0
        self.time_after_bidding = 0
        self.session_id = None
        self.feedback_count = 0
        self.goal_count = 0
        self.map_updating = True
        # communinication session variables end here
        self.intersections_requested = False
        self.exploration_time = rospy.Time.now().to_sec()
        self.last_evaluation_time = rospy.Time.now().to_sec()
        self.candidate_robots = self.frontier_robots + self.base_stations
        self.rate = rospy.Rate(0.1)
        self.run = rospy.get_param("/run")
        self.termination_metric = rospy.get_param("/termination_metric")
        self.max_exploration_time = rospy.get_param("/max_exploration_time")
        self.max_coverage = rospy.get_param("/max_coverage")
        self.max_common_coverage = rospy.get_param("/max_common_coverage")
        self.max_wait_time = rospy.get_param("/max_wait_time")
        self.environment = rospy.get_param("/environment")
        self.robot_count = rospy.get_param("/robot_count")
        self.debug_mode = rospy.get_param("/debug_mode")
        self.method = rospy.get_param("/method")
        # self.mac_id = rospy.get_param("/mac_id")
        self.comm_range = rospy.get_param("/comm_range")

        self.first_message_sent = False
        self.sent_messages = []
        self.received_messages = []
        self.newest_msg_ts = 0
        self.received_msgs_ts = {}
        self.last_saved_msg_ts = {}

        self.buff_data_srv = rospy.Service('/robot_{}/shared_data'.format(self.robot_id), SharedData,
                                           self.shared_data_handler)
        self.auction_points_srv = rospy.Service("/robot_{}/auction_points".format(self.robot_id), SharedPoint,
                                                self.shared_point_handler)
        self.alloc_point_srv = rospy.Service("/robot_{}/allocated_point".format(self.robot_id), SharedFrontier,
                                             self.shared_frontier_handler)
        rospy.Service('/robot_{}/gate_change_alert'.format(self.robot_id), GateChange, self.gate_change_handler)
        rospy.Subscriber('/robot_{}/initial_data'.format(self.robot_id), BufferedData,
                         self.initial_data_callback)  # just for initial data *
        self.karto_pub = rospy.Publisher("/robot_{}/karto_in".format(self.robot_id), LocalizedScan, queue_size=10)
        self.signal_strength_srv = rospy.ServiceProxy("/signal_strength".format(self.robot_id), HotSpot)
        self.signal_strength_srv.wait_for_service()
        self.fetch_frontier_points = rospy.ServiceProxy('/robot_{}/frontier_points'.format(self.robot_id),
                                                        FrontierPoint)

        self.graph_distance_srv = rospy.ServiceProxy('/robot_{}/graph_distance'.format(self.robot_id),
                                                     GraphDistance)

        self.check_intersections = rospy.ServiceProxy('/robot_{}/check_intersections'.format(self.robot_id),
                                                      Intersections)
        rospy.Subscriber('/robot_{}/gvgexplore/feedback'.format(self.robot_id), Pose, self.explore_feedback_callback)
        self.data_size_pub = rospy.Publisher('/shared_data_size', DataSize, queue_size=10)
        for rid in self.candidate_robots:
            received_data_clt = rospy.ServiceProxy("/robot_{}/shared_data".format(rid), SharedData)
            action_points_clt = rospy.ServiceProxy("/robot_{}/auction_points".format(rid), SharedPoint)
            alloc_point_clt = rospy.ServiceProxy("/robot_{}/allocated_point".format(rid), SharedFrontier)
            gate_change_clt = rospy.ServiceProxy("/robot_{}/gate_change_alert".format(rid), GateChange)
            pub = rospy.Publisher("/robot_{}/initial_data".format(rid), BufferedData, queue_size=10)
            self.publisher_map[rid] = pub
            self.allocation_pub[rid] = alloc_point_clt
            self.change_gate_alert[rid] = gate_change_clt
            self.shared_data_srv_map[rid] = received_data_clt
            self.shared_point_srv_map[rid] = action_points_clt
        rospy.Subscriber('/karto_out'.format(self.robot_id), LocalizedScan, self.robots_karto_out_callback,
                         queue_size=10)
        self.is_shutdown_caller = False

        self.trans_matrices = {}
        self.inverse_trans_matrices = {}
        self.prev_intersects = []
        self.poses_to_next_leaf = []

        # ======= pose transformations====================
        self.listener = tf.TransformListener()
        self.gvgexplore_goal_pub = rospy.Publisher('/robot_{}/gvgexplore/goal'.format(self.robot_id), Frontier,
                                                   queue_size=1)
        self.goal_cancel_srv = rospy.ServiceProxy('/robot_{}/gvgexplore/cancel'.format(self.robot_id),
                                                  CancelExploration)
        self.current_gate_srv = rospy.ServiceProxy('/robot_{}/gvgexplore/gate_leaf'.format(self.robot_id), CurrentGate)
        self.change_gate_srv = rospy.ServiceProxy('/robot_{}/gvgexplore/change_gate'.format(self.robot_id), GateChange)

        rospy.Subscriber('/robot_{}/gvgexplore/idle'.format(self.robot_id), Pose, self.idle_callback)
        rospy.Subscriber("/robot_{}/gvgexplore/feedback".format(self.robot_id), Pose, self.explore_feedback_callback)
        rospy.Subscriber("/shutdown", String, self.save_all_data)
        rospy.Subscriber("intersection", Pose, self.check_data_sharing_status)
        rospy.loginfo("Robot {} Initialized successfully!!".format(self.robot_id))

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            try:
                pu.log_msg(self.robot_id, "Is exploring: {}, Session ID: {}".format(self.is_exploring, self.session_id),
                           self.debug_mode)
            except Exception as e:
                pu.log_msg(self.robot_id, "Throwing error: {}".format(e), self.debug_mode)
            r.sleep()

    def check_data_sharing_status(self, data):
        if self.is_exploring:
            pu.log_msg(self.robot_id, "Intersec callback, session: {}".format(self.session_id), self.debug_mode)
            close_devices = self.get_close_devices()
            if not self.session_id:  # devices available and you're not in session
                pu.log_msg(self.robot_id, "Before calling intersection: {}".format(self.session_id), self.debug_mode)
                self.send_data(close_devices)
        else:
            pu.log_msg(self.robot_id, "Can't communicate. Robot not exploring", self.debug_mode)

    def send_data(self, current_devices):
        pu.log_msg(self.robot_id, "RECEIVED INTERSECTION CALLBACK", self.debug_mode)
        # self.cancel_exploration()
        self.is_sender = True
        self.session_id = '{}_{}'.format(self.robot_id, rospy.Time.now().to_sec())
        session_devices = []
        buff_data = {}
        current_gates = {int(self.robot_id): self.current_gate_srv().gate_leaf}
        local_data_size = 0
        for rid in current_devices:
            message_data = self.load_data_for_id(rid)
            local_data_size += len(message_data)  # self.get_message_size(message_data)
            buffered_data = self.create_buffered_data_msg(message_data, self.session_id, rid)
            response = self.shared_data_srv_map[rid](SharedDataRequest(req_data=buffered_data))
            pu.log_msg(self.robot_id, "received feedback from robot: {}".format(response.in_session), self.debug_mode)
            if not response.in_session:
                session_devices.append(rid)
                buff_data[rid] = response.res_data
                current_gates[int(rid)] = response.gate_leaf
                self.delete_data_for_id(rid)
            else:
                pu.log_msg(self.robot_id, "Robot {} is in another session".format(rid), self.debug_mode)
        self.process_data(buff_data, session_id=self.session_id, sent_data=local_data_size)
        self.decide_on_next_goal(current_gates)

    def decide_on_next_goal(self, current_gates):
        common_gates = []
        for i, p1 in current_gates.items():
            for j, p2 in current_gates.items():
                if i != j and (j, i) not in common_gates and p1 == p2:
                    common_gates.append((i, j))

        alerted_robots = []
        pu.log_msg(self.robot_id,
                   "Current gates: {}, Common gates: {}".format(current_gates, common_gates),
                   1 - self.debug_mode)
        for c in common_gates:
            if c[0] not in alerted_robots and c[1] not in alerted_robots:
                flagged_rid = current_gates[c[0]]
                if c[0] == self.robot_id:
                    self.change_gate_srv(GateChangeRequest(flagged_gates=[flagged_rid]))
                else:
                    self.change_gate_alert[str(c[0])](GateChangeRequest(flagged_gates=[flagged_rid]))
                alerted_robots.append(c[0])

        if self.robot_id not in alerted_robots:
            self.change_gate_srv(GateChangeRequest(flagged_gates=[]))

        pu.log_msg(self.robot_id, "Simillar gates: {}, Alerted robots: {}".format(common_gates, alerted_robots),
                   1 - self.debug_mode)
        self.session_id = None
        self.is_sender = False

    #
    def idle_callback(self, pose):
        pu.log_msg(self.robot_id, "Assigned region is complete. Robot is idle", 1 - self.debug_mode)

    def hungarian_assignment(self, bgraph, frontiers):
        rids = list(bgraph)
        rids.sort()
        rcosts = []
        for i in rids:
            rcosts.append(bgraph[i])
        cost = np.array(rcosts)
        row_ind, col_ind = linear_sum_assignment(cost)
        assignments = {}
        for i in range(len(row_ind)):
            if i != self.robot_id:
                frontier = self.create_frontier(i, col_ind[i], frontiers)
                res = self.allocation_pub[str(i)](SharedFrontierRequest(frontier=frontier))
            assignments[i] = col_ind[i]
        rospy.logerr("Do hungarian assignment: {}".format(assignments))
        return assignments

    def handle_intersection(self, current_devices):
        # pu.log_msg(self.robot_id, "RECEIVED INTERSECTION CALLBACK",self.debug_mode)
        self.cancel_exploration()
        self.is_sender = True
        self.session_id = '{}_{}'.format(self.robot_id, rospy.Time.now().to_sec())
        session_devices = []
        buff_data = {}
        local_data_size = 0
        for rid in current_devices:
            message_data = self.load_data_for_id(rid)
            local_data_size += len(message_data)  # self.get_message_size(message_data)
            buffered_data = self.create_buffered_data_msg(message_data, self.session_id, rid)
            response = self.shared_data_srv_map[rid](SharedDataRequest(req_data=buffered_data))
            pu.log_msg(self.robot_id, "received feedback from robot: {}".format(response.in_session), self.debug_mode)
            if not response.in_session:
                session_devices.append(rid)
                buff_data[rid] = response.res_data
                self.delete_data_for_id(rid)
            else:
                pu.log_msg(self.robot_id, "Robot {} is in another session".format(rid), self.debug_mode)
        self.process_data(buff_data, session_id=self.session_id, sent_data=local_data_size)
        frontier_point_response = self.fetch_frontier_points(
            FrontierPointRequest(count=len(current_devices) + 1, current_paths=[]))
        frontier_points = frontier_point_response.frontiers
        robot_assignments = {}
        if frontier_points:
            distances = self.compute_distances(self.get_robot_pose(), frontier_points)
            bgraph = {self.robot_id: [distances[i] for i in range(len(frontier_points))]}
            if session_devices:
                auction = self.create_auction(frontier_points, self.session_id)
                pu.log_msg(self.robot_id, "session devices: {}".format(session_devices), self.debug_mode)
                for rid in session_devices:
                    pu.log_msg(self.robot_id, "Action Request to Robot {}".format(rid), self.debug_mode)
                    auction_response = self.shared_point_srv_map[rid](SharedPointRequest(req_data=auction))
                    if auction_response.auction_accepted == 1:
                        data = auction_response.res_data
                        bgraph[int(rid)] = [data.distances[i] for i in range(len(data.distances))]

                robot_assignments = self.hungarian_assignment(bgraph, frontier_points)
            else:
                pu.log_msg(self.robot_id, "All robots are busy..", self.debug_mode)
        else:
            pu.log_msg(self.robot_id, "Just send an empty message so that the robots move on", self.debug_mode)
            for rid in current_devices:
                frontier = self.create_frontier(rid, [])
                res = self.allocation_pub[rid](SharedFrontierRequest(frontier=frontier))

        pu.log_msg(self.robot_id, "Point allocation complete", self.debug_mode)

        self.start_exploration_action(robot_assignments[self.robot_id], frontier_points)
        self.is_sender = False
        self.all_feedbacks.clear()
        # =============Ends here ==============

    def compute_next_frontier(self, taken_poses, frontier_points):
        ridge = None
        robot_pose = self.get_robot_pose()
        dist_dict = {}
        for point in frontier_points:
            dist_dict[pu.D(robot_pose, [point.position.x, point.position.y])] = point
        while not ridge and dist_dict:
            min_dist = min(list(dist_dict))
            closest = dist_dict[min_dist]
            if closest not in taken_poses:
                ridge = closest
                break
            del dist_dict[min_dist]
        return ridge

    def explore_feedback_callback(self, data):
        if not self.is_exploring:
            self.is_exploring = True
            self.session_id = None
            self.intersections_requested = False
            pu.log_msg(self.robot_id, "Received goal feedback", self.debug_mode)

    def gate_change_handler(self, req):
        res = self.change_gate_srv(GateChangeRequest(flagged_gates=req.flagged_gates)).result
        return GateChangeResponse(result=res)

    def shared_frontier_handler(self, req):
        data = req.frontier
        if not self.is_sender and data.session_id == self.session_id:
            other_frontiers = []
            for fp in data.frontiers:
                fp_in_sender = PoseStamped()
                fp_in_sender.header = data.msg_header.header
                fp_in_sender.pose = fp
                other_frontiers.append(
                    self.listener.transformPose("robot_{}/map".format(self.robot_id), fp_in_sender).pose)
            self.start_exploration_action(data.frontier_id, other_frontiers)
        pu.log_msg(self.robot_id, "Received allocated points", self.debug_mode)
        return SharedFrontierResponse(success=1)

    def create_frontier(self, receiver, frontier_id, frontier_points):
        frontier = Frontier()
        frontier.msg_header.header.frame_id = 'robot_{}/map'.format(self.robot_id)
        frontier.msg_header.header.stamp = rospy.Time.now()
        frontier.msg_header.sender_id = str(self.robot_id)
        frontier.msg_header.receiver_id = str(receiver)
        frontier.msg_header.topic = 'allocated_point'
        frontier.frontier_id = frontier_id  #
        frontier.frontiers = frontier_points
        frontier.session_id = self.session_id
        return frontier

    def create_auction(self, rendezvous_poses, session_id, distances=[]):
        auction = Auction()
        auction.msg_header.header.frame_id = 'robot_{}/map'.format(self.robot_id)
        auction.msg_header.sender_id = str(self.robot_id)
        auction.msg_header.topic = 'auction_points'
        auction.msg_header.header.stamp = rospy.Time.now()
        if not session_id:
            auction.session_id = ''
        auction.session_id = session_id
        auction.distances = distances
        auction.poses = []
        for f in rendezvous_poses:
            auction.poses.append(f)
        return auction

    def robots_karto_out_callback(self, data):
        if data.robot_id - 1 == self.robot_id:
            pu.log_msg(self.robot_id,
                       "ROBOT received a message Robot is saving a karto message: {}".format(data.robot_id),
                       self.debug_mode)
            for rid in self.candidate_robots:
                self.add_to_file(rid, [data])
            if self.is_initial_data_sharing:
                self.push_messages_to_receiver(self.candidate_robots, None, initiator=1)
                self.is_initial_data_sharing = False

    def push_messages_to_receiver(self, receiver_ids, session_id, is_alert=0, initiator=0):
        for receiver_id in receiver_ids:
            message_data = self.load_data_for_id(receiver_id)
            buffered_data = self.create_buffered_data_msg(message_data, session_id, receiver_id)
            self.publisher_map[str(receiver_id)].publish(buffered_data)
        # self.delete_data_for_id(receiver_id)

    def create_buffered_data_msg(self, message_data, session_id, receiver_id):
        buffered_data = BufferedData()
        if session_id:
            buffered_data.session_id = session_id
        else:
            buffered_data.session_id = ''
        buffered_data.msg_header.header.frame_id = '{}'.format(self.robot_id)
        buffered_data.msg_header.sender_id = str(self.robot_id)
        buffered_data.msg_header.receiver_id = str(receiver_id)
        buffered_data.msg_header.topic = 'initial_data'
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

    def save_message(self, karto):
        rid = karto.robot_id - 1
        ts = karto.scan.header.stamp.to_sec()
        should_save = False
        if rid not in self.received_msgs_ts:
            self.received_msgs_ts[rid] = ts
            should_save = True
        else:
            if self.received_msgs_ts[rid] < ts:
                self.received_msgs_ts[rid] = ts
                should_save = True
        if should_save:
            self.add_to_file(rid, [karto])
        return should_save

    def process_data(self, buff_data, session_id=None, sent_data=0):
        # rospy.logerr("data to process: {}".format(buff_data))
        # self.lock.acquire()
        self.map_updating = True
        for rid, rdata in buff_data.items():
            data_vals = rdata.data
            for scan in data_vals:
                if self.save_message(scan):
                    self.karto_pub.publish(scan)
            sent_data += len(data_vals)
        self.map_updating = False
        if self.is_sender:
            data_size = DataSize()
            data_size.header.frame_id = 'robot_{}/map'.format(self.robot_id)
            data_size.header.stamp = rospy.Time.now()
            data_size.size = sent_data
            data_size.session_id = session_id
            self.data_size_pub.publish(data_size)
        # self.lock.release()

    def get_message_size(self, msgs):
        size = 0
        for m in msgs:
            size += 1.0  # sys.getsizeof(m)
        return size

    def shared_data_handler(self, data):
        if self.session_id:
            return SharedDataResponse(in_session=1)
        buff_data = data.req_data
        received_data = {buff_data.msg_header.sender_id: buff_data}
        sender_id = buff_data.msg_header.sender_id
        session_id = buff_data.session_id
        message_data = self.load_data_for_id(sender_id)
        self.session_id = session_id
        self.process_data(received_data)
        mynext_goal_path = []
        # if self.is_exploring:
        #     self.cancel_exploration()
        buff_data = self.create_buffered_data_msg(message_data, session_id, sender_id)
        self.delete_data_for_id(sender_id)
        if self.is_exploring:
            self.session_id = None
        gate_leaf = self.current_gate_srv().gate_leaf
        return SharedDataResponse(in_session=0, res_data=buff_data, gate_leaf=gate_leaf)

    def shared_point_handler(self, auction_data):
        pu.log_msg(self.robot_id, "Received auction", self.debug_mode)
        data = auction_data.req_data
        session_id = data.session_id
        if self.is_sender or not self.session_id or session_id != self.session_id:
            return SharedPointResponse(auction_accepted=0)
        pu.log_msg(self.robot_id, "creating auction response", self.debug_mode)
        sender_id = data.msg_header.sender_id
        poses = data.poses
        if not poses and self.frontier_ridge:
            pu.log_msg(self.robot_id, "No poses received. Proceeding to my next frontier", self.debug_mode)
            self.start_exploration_action(self.frontier_ridge)
            return SharedPointResponse(auction_accepted=1, res_data=None)

        # convert received poses into this robot's reference frame and get its topological distances
        received_points = []
        p_in_sender = PoseStamped()
        p_in_sender.header = data.msg_header.header
        for p_from_sender in poses:
            p_in_sender.pose = p_from_sender
            p = self.listener.transformPose("robot_{}/map".format(self.robot_id), p_in_sender)
            pose = Pose()
            pose.position.x = p.pose.position.x
            pose.position.y = p.pose.position.y
            received_points.append(pose)
        distances = self.compute_distances(self.get_robot_pose(), received_points)
        auction = self.create_auction(poses, data.session_id, distances)
        pu.log_msg(self.robot_id, "sending auction data back", self.debug_mode)
        return SharedPointResponse(auction_accepted=1, res_data=auction)

    def initial_data_callback(self, buff_data):
        sender_id = buff_data.msg_header.sender_id
        self.process_data({sender_id: buff_data})
        # ============ used during initialization ============
        if self.initial_receipt:
            self.initial_data.add(sender_id)
            pu.log_msg(self.robot_id, "Counts: {}".format(len(self.initial_data)), self.debug_mode)
            if len(self.initial_data) == len(self.candidate_robots):
                self.initial_receipt = False
                if self.i_have_least_id():
                    self.is_sender = True
                    self.session_id = '{}_{}'.format(self.robot_id, rospy.Time.now().to_sec())
                    self.comm_session_time = rospy.Time.now().to_sec()
                    self.waiting_for_response = True
                    close_devices = self.get_close_devices()
                    pu.log_msg(self.robot_id, "Close devices {}:".format(close_devices), self.debug_mode)
                    if close_devices:
                        self.handle_intersection(close_devices)
                else:
                    pu.log_msg(self.robot_id, "Waiting for frontier points...", self.debug_mode)

    def compute_distances(self, robot_pose, target_poses):
        rpose = Pose()
        rpose.position.x = robot_pose[0]
        rpose.position.y = robot_pose[1]
        self.graph_distance_srv.wait_for_service()
        distances = self.graph_distance_srv(GraphDistanceRequest(source=rpose, targets=target_poses)).distances
        return distances

    def start_exploration_action(self, frontier_id, other_leaves):
        while self.map_updating:  # wait for map to update
            sleep(1)
        self.feedback_count = 0
        front = Frontier()
        front.frontier_id = frontier_id
        front.frontiers = other_leaves
        self.gvgexplore_goal_pub.publish(front)

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
            if True:
                self.listener.waitForTransform("robot_{}/map".format(self.robot_id),
                                               "robot_{}/base_link".format(self.robot_id), rospy.Time(),
                                               rospy.Duration(4.0))
                (robot_loc_val, rot) = self.listener.lookupTransform("robot_{}/map".format(self.robot_id),
                                                                     "robot_{}/base_link".format(self.robot_id),
                                                                     rospy.Time(0))
                robot_pose = (math.floor(robot_loc_val[0]), math.floor(robot_loc_val[1]), robot_loc_val[2])
                pu.log_msg(self.robot_id, "Robot pose: {}".format(robot_pose), self.debug_mode)
            # except: # TODO: catch proper exceptions.
            #    pass
        return robot_pose

    def cancel_exploration(self):
        self.intersections_requested = True
        if self.is_exploring:
            try:
                self.poses_to_next_leaf = self.goal_cancel_srv(CancelExplorationRequest(req=1)).path_to_next_leaf
                pu.log_msg(self.robot_id, "Canceling exploration ...", self.debug_mode)
            except:
                pu.log_msg(self.robot_id, "Error while cancelling..", self.debug_mode)
            self.is_exploring = False
            # return poses_to_next_leaf

    def add_to_file(self, rid, data):
        # self.lock.acquire()
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
        # self.lock.acquire()
        if rid in self.karto_messages:
            del self.karto_messages[rid]
        # self.lock.release()
        return True

    def theta(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        if dx == 0:
            dx = 0.000001
        return math.atan2(dy, dx)

    def D(self, p, q):
        pu.log_msg(self.robot_id, "Params: {}, {}".format(p, q), self.debug_mode)
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def get_elevation(self, quaternion):
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        return yaw

    def save_all_data(self, data):
        rospy.signal_shutdown("Shutting down Robot")


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
