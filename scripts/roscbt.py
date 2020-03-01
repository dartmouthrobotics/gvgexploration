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
        # performance datastructures end here
        self.publisher_map = {}
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

        self.termination_metric = rospy.get_param("/robot_0/0/node0/termination_metric")
        self.robot_count = rospy.get_param("/robot_0/0/node0/robot_count")
        self.environment = rospy.get_param("/robot_0/0/node0/environment")
        self.run = rospy.get_param("/robot_0/0/node0/run")

        # difference in center of map image and actual simulation
        self.dx = self.world_center[0] - self.map_pose[0]
        self.dy = self.world_center[1] - self.map_pose[1]
        self.exploration_data = []

        # import message types
        for topic in self.topics:
            msg_pkg = topic["message_pkg"]
            msg_type = topic["message_type"]
            topic_name = topic["name"]
            exec("from {}.msg import {}\n".format(msg_pkg, msg_type))
            # rospy.logerr("from {}.msg import {}\n".format(msg_pkg, msg_type))
            # creating publishers data structure
            self.publisher_map[topic_name] = {}

        self.pose_desc = {}
        self.explored_area = {}
        self.coverage = {}
        self.connected_robots = {}
        for i in self.robot_ids:
            exec('self.signal_pub[{0}]=rospy.Publisher("/roscbt/robot_{0}/signal_strength", SignalStrength,'
                 'queue_size=10)'.format(i))
            if str(i) in self.shared_topics:
                topic_map = self.shared_topics[str(i)]
                for id, topic_dict in topic_map.items():
                    for k, v in topic_dict.items():
                        exec("def {0}_{1}_{2}(self, data):self.main_callback({1},{2},data,'{0}')".format(k, id, i))
                        exec("setattr(roscbt, '{0}_callback{1}_{2}', {0}_{1}_{2})".format(k, id, i))
                        exec(
                            "rospy.Subscriber('/robot_{1}/{2}', {3}, self.{2}_callback{0}_{1}, queue_size = 100)".format(
                                id, i, k, v))
                        # populating publisher datastructure
                        exec('pub=rospy.Publisher("/roscbt/robot_{}/{}", {}, queue_size=10)'.format(i, k, v))
                        if i not in self.publisher_map[k]:
                            exec('self.publisher_map["{}"]["{}"]=pub'.format(k, i))

        # ======= pose transformations====================
        self.robot_pose = {}
        self.prev_poses = {}
        for i in self.robot_ids:
            s = "def a_" + str(i) + "(self, data): self.robot_pose[" + str(i) + "] = (data.pose.pose.position.x," \
                                                                                "data.pose.pose.position.y," \
                                                                                "data.pose.pose.position.z) "
            exec(s)
            exec("setattr(roscbt, 'callback_pos_teammate" + str(i) + "', a_" + str(i) + ")")
            exec("rospy.Subscriber('/robot_" + str(
                i) + "/base_pose_ground_truth', Odometry, self.callback_pos_teammate" + str(i) + ", queue_size = 100)")

        # self.listener = tf.TransformListener()
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)
        self.already_shutdown = False
        rospy.loginfo("ROSCBT Initialized Successfully!")

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            try:
                self.share_signal_strength()
                self.get_coverage()
                self.compute_performance()
                r.sleep()
                # time.sleep(10)
            except Exception as e:
                rospy.logerr('interrupted!: {}'.format(e))
                break

    def read_map_image(self, image_path):
        im = Image.open(image_path, 'r')
        pix_val = list(im.getdata())
        size = im.size
        pixel_values = {}
        for index in range(len(pix_val)):
            i = int(np.floor(index / size[0]))
            j = index % size[0]
            pixel_values[(i, j)] = pix_val[index][0]

        return size, pixel_values

    ''' Resolving robot ids should be customized by the developer  '''

    def resolve_sender(self, robot_id1, topic, data):
        sender_id = None
        if topic == 'received_data':
            sender_id = data.header.frame_id
        elif topic == 'auction_points':
            sender_id = data.header.frame_id
        if sender_id == str(robot_id1):
            return sender_id

    def resolve_receiver(self, robot_id2, topic, data):
        return str(robot_id2)

    def main_callback(self, robot_id1, robot_id2, data, topic):
        current_time = rospy.Time.now().secs
        sender_id = data.header.frame_id  # self.resolve_sender(robot_id1, topic, data)
        receiver_id = self.resolve_receiver(robot_id2, topic, data)
        if sender_id and receiver_id:
            combn = (sender_id, receiver_id)
            # handle all message types
            distance, in_range = self.can_communicate(sender_id, receiver_id)
            if combn in self.distances:
                self.distances[combn][current_time] = distance
            else:
                self.distances[combn] = {current_time: distance}
            if in_range:
                # rospy.logerr("Data sent from {} to {} on topic: {}".format(receiver_id, sender_id, topic))
                self.publisher_map[topic][receiver_id].publish(data)

                data_size = sys.getsizeof(data)
                if combn in self.sent_data:
                    self.sent_data[combn][current_time] = data_size
                else:
                    self.sent_data[combn] = {current_time: data_size}
            else:
                rospy.logerr(
                    "Robot {} and {} are out of range topic {}: {} m".format(receiver_id, sender_id, topic, distance))

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
                        self.signal_pub[int(r1)].publish(signal_strength)

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
            data = {'start_time':  rospy.Time.now().to_sec()}

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
            self.lasttime_before_performance_calc =  rospy.Time.now().to_sec()
        except Exception as e:
            rospy.logerr("getting error: {}".format(e))
        finally:
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

    def pixel2pose(self, point, origin_x, origin_y, resolution):
        new_p = [0.0] * 2
        new_p[INDEX_FOR_Y] = round(origin_x + point[INDEX_FOR_X] * resolution, 2)
        new_p[INDEX_FOR_X] = round(origin_y + point[INDEX_FOR_Y] * resolution, 2)
        return tuple(new_p)

    def map_update_callback(self, occ_grid):
        current_time = rospy.Time.now().secs
        resolution = occ_grid.info.resolution
        origin_pos = occ_grid.info.origin.position
        origin_x = origin_pos.x
        origin_y = origin_pos.y
        height = occ_grid.info.height
        width = occ_grid.info.width
        grid_values = np.array(occ_grid.data).reshape((height, width)).astype(
            np.float32)

        for row in range(height):
            for col in range(width):
                index = [0] * 2
                index[INDEX_FOR_X] = col
                index[INDEX_FOR_Y] = row
                index = tuple(index)
                pose = self.pixel2pose(index, origin_x, origin_y, resolution)
                p = grid_values[row, col]
                self.pose_desc[pose] = p
        explored = self.get_explored_area()
        self.explored_area[current_time] = explored

    def get_explored_area(self):
        total_area = len(self.pose_desc)
        all_poses = list(self.pose_desc)
        explored_area = 0
        for p in all_poses:
            if p != -1:
                explored_area += 1
        return total_area - explored_area

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
        save_data(self.exploration_data,'gvg/exploration_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,self.termination_metric))



if __name__ == '__main__':
    cbt = roscbt()
    cbt.spin()
