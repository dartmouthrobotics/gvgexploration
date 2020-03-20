#!/usr/bin/python
import pickle
from os import path
import copy
import rospy
import tf
import math
from PIL import Image
import numpy as np
from threading import Lock
import json
from gvgexploration.msg import SignalStrength, RobotSignal
from time import sleep
from nav_msgs.msg import OccupancyGrid
import sys
from project_utils import save_data
from std_msgs.msg import String
from threading import Thread

'''
ROS communication benchmarking tool (ROSCBT) is a simulator of a communication link between communication devices. 
ROSCBT is ROS node that is implemeted by this class, using settings that are specified by the user in the configuration file

All parameters are read from package/param/roscbt_config.yaml file that is added to your package

For this node to run, the following parameters MUST be available:
- robot_ids: IDs of robots that make up the multirobot communication network
- robot_map: all the various interconnections between robots as defined in the configuration file
- topics: all the topics through which the robots are exchanging messages
- topic remaps: a list of topics that have been remapped before use


@Author: Kizito Masaba and Alberto Quattrini Li
'''

# ids to identify the robot type
BS_TYPE = 1
RR_TYPE = 2
FR_TYPE = 3

WIFI_RANGE = 10
BLUETOOTH_RANGE = 5
MIN_SIGNAL_STRENGTH = -1 * 65.0

INDEX_FOR_X = 1
INDEX_FOR_Y = 0


class roscbt:
    def __init__(self):
        # data structures to store incoming messages that are pending forwarding

        # performance data structures
        self.ss = {}
        self.bandwidth = {}
        self.distances = {}
        self.tp = {}
        self.publish_time = {}
        self.sent_data = {}
        self.lock = Lock()
        self.communication_model = 1
        self.constant_distance_model = self.compute_constant_distance_ss()
        self.publisher_map = {}
        self.subsciber_map = {}
        self.signal_pub = {}
        rospy.init_node('roscbt', anonymous=True)
        self.lasttime_before_performance_calc = rospy.Time.now().to_sec()

        # we can load the map as an image to determine the location of obstacles in the environment
        self.map_topic = rospy.get_param("map_topic", '')
        self.robot_ids = rospy.get_param("/roscbt/robot_ids", [])
        self.robot_ranges = rospy.get_param("/roscbt/robot_ranges", {})
        self.topics = rospy.get_param("/roscbt/topics", [])

        # new parameter -- specifying how information is shared among robots
        self.shared_topics = rospy.get_param("/roscbt/shared_topics", {})

        # processing groundtruth about the map
        map_image_path = rospy.get_param("/roscbt/map_image_path", '')
        self.world_scale = rospy.get_param("/roscbt/world_scale", 1)
        self.map_pose = rospy.get_param("/roscbt/map_pose", [])
        self.world_center = rospy.get_param("/roscbt/world_center", [])

        self.termination_metric = rospy.get_param("~termination_metric")
        self.robot_count = rospy.get_param("~robot_count")
        self.environment = rospy.get_param("~environment")
        self.run = rospy.get_param("~run")

        # difference in center of map image and actual simulation
        self.dx = self.world_center[0] - self.map_pose[0]
        self.dy = self.world_center[1] - self.map_pose[1]
        self.exploration_data = []
        self.sent_messages = []
        self.received_messages = []

        # import message types
        for topic in self.topics:
            msg_pkg = topic["message_pkg"]
            msg_type = topic["message_type"]
            topic_name = topic["name"]
            exec("from {}.msg import {}\n".format(msg_pkg, msg_type))
            # rospy.logerr("from {}.msg import {}\n".format(msg_pkg, msg_type))
            # creating publishers data structure
            self.publisher_map[topic_name] = {}
            self.subsciber_map[topic_name] = {}

        self.pose_desc = {}
        self.coverage = {}
        self.connected_robots = {}

        for receiver_id in self.robot_ids:
            sig_pub = rospy.Publisher("/robot_{0}/signal_strength".format(receiver_id), SignalStrength, queue_size=10)
            self.signal_pub[receiver_id] = sig_pub
            if str(receiver_id) in self.shared_topics:
                topic_map = self.shared_topics[str(receiver_id)]
                for sender_id, topic_dict in topic_map.items():
                    for topic_name, topic_type in topic_dict.items():
                        if sender_id not in self.subsciber_map[topic_name]:
                            sub = None
                            exec("sub=rospy.Subscriber('/roscbt/robot_{0}/{2}', {3}, self.main_callback,queue_size=10)".format(sender_id, receiver_id, topic_name, topic_type))
                            self.subsciber_map[topic_name][sender_id] = sub
                        if receiver_id not in self.publisher_map[topic_name]:
                            pub = None
                            exec('pub=rospy.Publisher("/robot_{}/{}", {}, queue_size=10)'.format(receiver_id,topic_name,topic_type))
                            self.publisher_map[topic_name][receiver_id] = pub

        # ======= pose transformations====================
        self.robot_pose = {}
        self.prev_poses = {}
        for i in self.robot_ids:
            exec("def a_{0}(self, data): self.robot_pose[{0}] = (data.pose.pose.position.x,data.pose.pose.position.y,"
                 "(data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z,"
                 "data.pose.pose.orientation.w), data.header.stamp.to_sec())".format(i))
            exec("setattr(roscbt, 'callback_pos_teammate{0}', a_{0})".format(i))
            exec("rospy.Subscriber('/robot_{0}/base_pose_ground_truth', Odometry, self.callback_pos_teammate{0}, "
                 "queue_size = 100)".format(i))

            # self.listener = tf.TransformListener()
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)
        self.already_shutdown = False
        rospy.loginfo("ROSCBT Initialized Successfully!")

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            # try:
            self.share_signal_strength()
            self.get_coverage()
            self.compute_performance()
            r.sleep()
            # time.sleep(10)
            # except Exception as e:
            #     rospy.logerr('interrupted!: {}'.format(e))
            #     break
    #
    # def main_callback(self, data):
    #     thread = Thread(target=self.handle_request, args=(data,))
    #     thread.daemon = True
    #     thread.start()

    def main_callback(self, data):
        sender_id = data.msg_header.sender_id
        receiver_id = data.msg_header.receiver_id
        topic = data.msg_header.topic
        start_time = rospy.Time.now().to_sec()
        self.received_messages.append(
            {'time': start_time, 'message_time': data.msg_header.header.stamp.to_sec(), 'sender_id': sender_id,
             'receiver_id': receiver_id, 'session_id': data.session_id.data, 'topic': topic})
        current_time = rospy.Time.now().secs
        combn = (sender_id, receiver_id)
        # handle all message types
        distance, in_range = self.can_communicate(sender_id, receiver_id)
        if combn in self.distances:
            self.distances[combn][current_time] = distance
        else:
            self.distances[combn] = {current_time: distance}
        if in_range:
            self.publisher_map[topic][receiver_id].publish(data)
            now = rospy.Time.now().secs
            time_diff = now - start_time
            self.sent_messages.append(
                {'time': now, 'message_time': data.msg_header.header.stamp.to_sec(), 'sender_id': sender_id,
                 'receiver_id': receiver_id, 'session_id': data.session_id.data, 'time_diff': time_diff,
                 'topic': topic})
            data_size = sys.getsizeof(data)
            if combn in self.sent_data:
                self.sent_data[combn][current_time] = data_size
            else:
                self.sent_data[combn] = {current_time: data_size}
           # rospy.logerr("Data sent from {} to {} on topic: {}".format(sender_id, receiver_id, topic))
        else:
            rospy.logerr("Robot {} and {} are out of range topic {}: {} m".format(receiver_id, sender_id, topic, distance))

    # method to check the constraints for robot communication
    def can_communicate(self, robot_id1, robot_id2):
        robot1_pose = self.get_robot_pose(robot_id1)
        robot2_pose = self.get_robot_pose(robot_id2)
        if not robot1_pose or not robot2_pose:
            return -1, False
        return self.robots_inrange(robot1_pose, robot2_pose)

    def share_signal_strength(self):
        for r1 in self.robot_ids:
            signal_strength = SignalStrength()
            rsignals = []
            for r2 in self.robot_ids:
                if r1 != r2:
                    robot1_pose = self.get_robot_pose(r1)
                    robot2_pose = self.get_robot_pose(r2)
                    if robot1_pose and robot2_pose:
                        d = math.floor(math.sqrt(
                            ((robot1_pose[0] - robot2_pose[0]) ** 2) + ((robot1_pose[1] - robot2_pose[1]) ** 2)))
                        ss = self.compute_signal_strength(d)
                        if ss >= MIN_SIGNAL_STRENGTH:
                            robot_signal = RobotSignal()
                            robot_signal.robot_id = int(r2)
                            robot_signal.rssi = ss
                            rsignals.append(robot_signal)
                        signal_strength.header.stamp = rospy.Time.now()
                        signal_strength.header.frame_id = 'roscbt'
                        signal_strength.signals = rsignals
                        self.signal_pub[r1].publish(signal_strength)

    def compute_constant_distance_ss(self):
        wifi_freq = 2.4 * math.pow(10, 9)
        c_light = 3 * math.pow(10, 8)
        wifi_wavelength = c_light / wifi_freq
        GL = math.sqrt(1)
        PI = math.pi
        constant_distance_model = (GL * wifi_wavelength) / PI
        return constant_distance_model

    def compute_signal_strength(self, d):
        if d < 0.001:
            return 0
        temp = self.constant_distance_model / d
        temp = temp * temp / 16
        rssi = 10 * math.log10(temp)
        return rssi

    def compute_performance(self):
        self.lock.acquire()
        try:
            current_time = rospy.Time.now().to_sec()
            data = {'start_time': rospy.Time.now().to_sec()}

            shared_data = []
            comm_ranges = []
            distances = []
            for sid in self.robot_ids:
                for rid in self.robot_ids:
                    if sid != rid:
                        combn = (sid, rid)
                        if combn in self.sent_data:
                            shared_data += [v for t, v in self.sent_data[combn].items() if
                                            self.lasttime_before_performance_calc < t <= current_time]

                        if combn in self.distances:
                            comm_ranges += [v for t, v in self.distances[combn].items() if
                                            self.lasttime_before_performance_calc < t <= current_time]

            if not comm_ranges:
                data['comm_ranges'] = [-1, -1]
            else:
                data['comm_ranges'] = [np.nanmean(comm_ranges), np.nanvar(comm_ranges)]
            if not shared_data:
                data['shared_data'] = [-1, -1]
            else:
                data['shared_data'] = [np.nanmean(shared_data), np.nanvar(shared_data)]

            coverage = [v for t, v in self.coverage.items() if
                        self.lasttime_before_performance_calc < t <= current_time]
            connected = [v for t, v in self.connected_robots.items() if
                         self.lasttime_before_performance_calc < t <= current_time]

            if not coverage:
                data['coverage'] = [-1, -1]
            else:
                data['coverage'] = [np.nanmean(coverage), np.nanvar(coverage)]

            if not connected:
                data['connected'] = [-1, -1]
            else:
                data['connected'] = [np.nanmean(connected), np.nanvar(connected)]

            robot_poses = copy.deepcopy(self.robot_pose)
            for rid, p in robot_poses.items():
                d = 0
                if rid in self.prev_poses:
                    d = self.D(self.prev_poses[rid], p)
                distances.append(d)
            data['distance'] = np.nansum(distances)
            self.prev_poses = robot_poses
            self.exploration_data.append(data)
            self.sent_data.clear()
            self.distances.clear()
            self.coverage.clear()
            self.connected_robots.clear()
            self.lasttime_before_performance_calc = rospy.Time.now().to_sec()
        except Exception as e:
            rospy.logerr("getting error: {}".format(e))
        finally:
            pass
        self.lock.release()

    '''
      computes euclidean distance between two cartesian coordinates
    '''

    def robots_inrange(self, loc1, loc2):
        distance = math.floor(math.sqrt(((loc1[0] - loc2[0]) ** 2) + ((loc1[1] - loc2[1]) ** 2)))
        return distance, True

    def get_robot_pose(self, robot_id):
        robot_pose = None
        if int(robot_id) in self.robot_pose:
            robot_pose = self.robot_pose[int(robot_id)]

        return robot_pose

    def D(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return math.sqrt(dx ** 2 + dy ** 2)

    def get_coverage(self):
        current_time = rospy.Time.now().secs
        distances = []
        connected = {}
        if self.robot_pose:
            for i in self.robot_ids:
                pose1 = self.get_robot_pose(i)
                if pose1:
                    for j in self.robot_ids:
                        if i != j:
                            pose2 = self.get_robot_pose(j)
                            if pose2:
                                distance, in_range = self.robots_inrange(pose1, pose2)
                                distances.append(distance)
                                if in_range:
                                    if i in connected:
                                        connected[i] += 1
                                    else:
                                        connected[i] = 1
        if not distances:
            result = [-1, -1]
        else:
            result = [np.nanmean(distances), np.nanstd(distances)]
        if connected:
            key = max(connected, key=connected.get)
            max_connected = connected[key] + 1
            self.connected_robots[current_time] = max_connected
        self.coverage[current_time] = result
        return result

    def shutdown_callback(self, msg):
        self.save_all_data()
        rospy.signal_shutdown('ROSCBT: Shutdown command received!')

    def save_all_data(self):
        save_data(self.exploration_data,
                  'gvg/exploration_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                              self.termination_metric))
        save_data(self.received_messages,
                  'gvg/roscbt_data_received_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                                       self.termination_metric))
        save_data(self.sent_messages,
                  'gvg/roscbt_data_sent_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                                   self.termination_metric))


if __name__ == '__main__':
    cbt = roscbt()
    cbt.spin()
