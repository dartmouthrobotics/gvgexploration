#!/usr/bin/python
import rospy
import tf
import math
from PIL import Image
import numpy as np
from threading import Lock
import json
from gvgexploration.msg import SignalStrength,RobotSignal
from time import sleep
import sys

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
MIN_SIGNAL_STRENGTH = -1*65.0

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
        self.signal_pub={}
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
        self.map_size, self.map_pixels = self.read_map_image(map_image_path)
        if not self.map_pixels:
            rospy.loginfo("File not found on path: {}".format(map_image_path))
            exit(1)
        self.world_scale = rospy.get_param("/roscbt/world_scale", 1)
        self.map_pose = rospy.get_param("/roscbt/map_pose", [])
        self.world_center = rospy.get_param("/roscbt/world_center", [])

        # difference in center of map image and actual simulation
        self.dx = self.world_center[0] - self.map_pose[0]
        self.dy = self.world_center[1] - self.map_pose[1]

        # import message types
        for topic in self.topics:
            msg_pkg = topic["message_pkg"]
            msg_type = topic["message_type"]
            topic_name = topic["name"]
            exec("from {}.msg import {}\n".format(msg_pkg, msg_type))
            # rospy.logerr("from {}.msg import {}\n".format(msg_pkg, msg_type))
            # creating publishers data structure
            self.publisher_map[topic_name] = {}

        for i in self.robot_ids:
            exec('self.signal_pub[{0}]=rospy.Publisher("/roscbt/robot_{0}/signal_strength", SignalStrength,'
                 'queue_size=10)'.format(i))
            if str(i) in self.shared_topics:
                topic_map = self.shared_topics[str(i)]
                for id, topic_dict in topic_map.items():
                    for k, v in topic_dict.items():
                        exec("def {0}_{1}_{2}(self, data):self.main_callback({1},{2},data,'{0}')".format(k, id, i))
                        exec("setattr(roscbt, '{0}_callback{1}_{2}', {0}_{1}_{2})".format(k, id, i))
                        exec("rospy.Subscriber('/robot_{1}/{2}', {3}, self.{2}_callback{0}_{1}, queue_size = 100)".format(id, i, k, v))
                        # populating publisher datastructure
                        exec('pub=rospy.Publisher("/roscbt/robot_{}/{}", {}, queue_size=10)'.format(i, k, v))
                        if i not in self.publisher_map[k]:
                            exec('self.publisher_map["{}"]["{}"]=pub'.format(k, i))

        # ======= pose transformations====================
        self.robot_pose={}
        for i in self.robot_ids:
            s = "def a_" + str(i) + "(self, data): self.robot_pose[" + str(i) + "] = (data.pose.pose.position.x," \
                                                                                "data.pose.pose.position.y," \
                                                                                "data.pose.pose.position.z) "
            exec(s)
            exec("setattr(roscbt, 'callback_pos_teammate" + str(i) + "', a_" + str(i) + ")")
            exec("rospy.Subscriber('/robot_" + str(i) + "/base_pose_ground_truth', Odometry, self.callback_pos_teammate" + str(i) + ", queue_size = 100)")

        # self.listener = tf.TransformListener()

        rospy.loginfo("ROSCBT Initialized Successfully!")

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            try:
                self.share_signal_strength()
                self.compute_performance()
                r.sleep()
                # time.sleep(10)
            except Exception as e:
                rospy.logerr('interrupted!: {}'.format(e))
                break

    '''
    Convert the world  pose into a pixel position in the image
    :param pose - the x,y coordinate in the map, which is mapped to a pixel in the actual image of the map
    '''

    def pose_to_pixel(self, pose):
        # rescale negative values to start from 0
        rescaled_world_pose_x = pose.x + self.world_center[0]
        rescaled_world_pose_y = pose.y + self.world_center[1]

        # convert world pose to pixel positions
        row = self.map_size[1] - (rescaled_world_pose_y + self.dy) * self.world_scale
        col = (rescaled_world_pose_x + self.dx) * self.world_scale
        pixel_pose = (row, col)
        return pixel_pose

    '''
     Convert a pixel position into pixel position into a world pose
     :param pixel_pose - pixel position that shall be converted into a world position
     :return pose - the pose in the world that corresponds to the given pixel position
    '''

    def pixel_to_pose(self, pixel_pose):
        rescaled_world_pose_x = (pixel_pose[1] / self.world_scale) - self.dx
        rescaled_world_pose_y = ((self.map_size[1] - pixel_pose[0]) / self.world_scale) - self.dy

        x = rescaled_world_pose_x - self.world_center[0]
        y = rescaled_world_pose_y - self.world_center[1]
        return (x, y)

    '''
     Get pixel positions of all the neighboring pixels
     :param pixel_pos - center pixel from which to get the neighbors
     :param distance - this defines the width of the neighborhood
     :return neighbors - all valid pixel positions that are neighboring the given pixel within the image

    '''

    def get_pixel_neighbors(self, pixel_pos, distance=1):
        x = pixel_pos[0]
        y = pixel_pos[1]
        neighbors = []
        pixels = []
        for i in range(1, distance + 1):
            east = (x, y + i)
            north = (x, y + i)
            west = (x - i, y)
            south = (x - 1, y)
            ne = (x + i, y + i)
            nw = (x - i, y + i)
            se = (x + i, y - i)
            sw = (x - i, y - i)
            possible_neigbors = [east, north, west, south, ne, nw, se, sw]
            for n in possible_neigbors:
                if (n[1], n[0]) in self.map_pixels:
                    neighbors.append(n)
                    pixels.append(self.map_pixels[n])
        return neighbors, pixels

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
        elif topic == 'rendezvous_points':
            sender_id = data.header.frame_id
        if sender_id == str(robot_id1):
            return sender_id

    def resolve_receiver(self, robot_id2, topic, data):
        return str(robot_id2)

    def main_callback(self, robot_id1, robot_id2, data, topic):
        current_time = rospy.Time.now().secs
        sender_id = self.resolve_sender(robot_id1, topic, data)
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
                self.publisher_map[topic][receiver_id].publish(data)

                data_size = sys.getsizeof(data)
                if combn in self.sent_data:
                    self.sent_data[combn][current_time] = data_size
                else:
                    self.sent_data[combn] = {current_time: data_size}
            else:
                rospy.logerr("Robot {} and {} are out of range: {} m".format(receiver_id, sender_id, distance))

    # method to check the constraints for robot communication
    def can_communicate(self, robot_id1, robot_id2):
        robot1_pose = self.get_robot_pose(robot_id1)
        robot2_pose = self.get_robot_pose(robot_id2)

        robot1_range = self.robot_ranges[robot_id1]
        robot2_range = self.robot_ranges[robot_id2]
        if not robot1_pose or not robot2_pose:
            return -1,False
        return self.robots_inrange(robot1_pose, robot2_pose, robot1_range, robot2_range)

    def share_signal_strength(self):
        for r1 in self.robot_ids:
            signal_strength=SignalStrength()
            rsignals=[]
            for r2 in self.robot_ids:
                if r1 !=r2:
                    robot1_pose = self.get_robot_pose(r1)
                    robot2_pose = self.get_robot_pose(r2)
                    if robot1_pose and robot2_pose:
                        d= math.floor(math.sqrt(((robot1_pose[0] - robot2_pose[0]) ** 2) + ((robot1_pose[1] - robot2_pose[1]) ** 2)))
                        ss= self.compute_signal_strength(d)
                        if ss >= MIN_SIGNAL_STRENGTH:
                            robot_signal = RobotSignal()
                            robot_signal.robot_id = int(r2)
                            robot_signal.rssi=ss
                            rsignals.append(robot_signal)
                        signal_strength.header.stamp=rospy.Time.now()
                        signal_strength.header.frame_id='roscbt'
                        signal_strength.signals=rsignals
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
            data = {'start_time': current_time}
            for sid in self.robot_ids:
                for rid in self.robot_ids:
                    if sid != rid:
                        combn = (sid, rid)
                        distance = {}
                        sd = {}
                        if combn in self.sent_data:
                            sd = {t: v for t, v in self.sent_data[combn].items() if
                                  self.lasttime_before_performance_calc < t <= current_time}
                        if combn in self.distances:
                            distance = {t: v for t, v in self.distances[combn].items() if
                                        self.lasttime_before_performance_calc < t <= current_time}
                        data['distance_{}_{}'.format(sid, rid)] = distance
                        data['sent_data_{}_{}'.format(sid, rid)] = sd

            json_data = json.dumps(data)

            with open('hrm_performance{}_{}.txt'.format(len(self.robot_ids), self.communication_model), 'a+') as fd:
                fd.write("{}\n".format(json_data))
            self.lasttime_before_performance_calc = current_time
        except Exception as e:
            rospy.logerr("getting error: {}".format(e.message))
        finally:
            self.lock.release()

    '''
     Determine the points through which the signal is transmitted from the transmitter to the receiver (only considering line of sight)
    '''

    def compute_signal_path(self, transmitter_id, receiver_id):
        tx_pose = self.robot_pose[transmitter_id]
        rx_pose = self.robot_pose[receiver_id]
        x1 = tx_pose.x
        y1 = tx_pose.y

        x2 = rx_pose.x
        y2 = rx_pose.y
        way_points = []
        dy = y2 - y1
        dx = x2 - y2
        m = dy / dx
        if m > 1:
            x1 = rx_pose.y
            y1 = rx_pose.x
            x2 = rx_pose.y
            y2 = rx_pose.x

            dx = x1 - x2
            if dx < 0:
                points = self.bresenham_path(x1, y1, x2, y2)
            else:
                points = self.bresenham_path(x2, y2, x1, y1)

            # flip the raster so that bresenham algorithm can work normally. PS: remember to flip the resulting
            # coordinates
            way_points += [(v[1], v[0]) for v in points]
        elif m < 1:
            dx = x1 - x2
            if dx < 0:
                way_points += self.bresenham_path(x1, y1, x2, y2)
            else:
                # reverse the points for the sake of bresenham algorithm
                way_points += self.bresenham_path(x2, y2, x1, y1)

        else:
            # just pick all points along the straight line since the gradient is 1
            dx = x1 - x2
            if dx < 0:
                way_points += [(v, v) for v in range(x1, x2 + 1)]
            else:
                way_points += [(v, v) for v in range(x2, x1 + 1)]

        return way_points

    '''
    Get all the points traversed by a line that connects 2 points
    '''

    def bresenham_path(self, x1, y1, x2, y2):
        points = []

        x = x1
        y = y1
        dx = x2 - x1
        dy = y2 - y1
        p = 2 * dx - dy
        while (x <= x2):
            points.append((x, y))
            x += 1
            if p < 0:
                p = p + 2 * dy
            else:
                p = p + 2 * dy - 2 * dx
                y += 1
        return points

    # computes euclidean distance between two cartesian coordinates
    def robots_inrange(self, loc1, loc2, robot1_range, robot2_range):
        distance = math.floor(math.sqrt(((loc1[0] - loc2[0]) ** 2) + ((loc1[1] - loc2[1]) ** 2)))
        return distance, True #distance <= robot1_range+1 and distance <= robot2_range

    def get_robot_pose(self, robot_id):
        robot_pose = None
        if int(robot_id) in self.robot_pose:
            robot_pose=self.robot_pose[int(robot_id)]
        # while not robot_pose:
        #     try:
        #         self.listener.waitForTransform("robot_{}/map".format(robot_id), "robot_{}/base_link".format(robot_id),
        #                                        rospy.Time(), rospy.Duration(4.0))
        #         (robot_pose_val, rot) = self.listener.lookupTransform("robot_{}/map".format(robot_id),
        #                                                               "robot_{}/base_link".format(robot_id),
        #                                                               rospy.Time(0))
        #
        #         robot_pose = (round(robot_pose_val[0], 1), round(robot_pose_val[1], 1))
        #         sleep(1)
        #     except:
        #         pass

        return robot_pose


if __name__ == '__main__':
    cbt = roscbt()
    cbt.spin()
