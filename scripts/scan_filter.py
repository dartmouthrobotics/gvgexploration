#!/usr/bin/python
import rospy
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer
import message_filters
from nav_msgs.msg import Odometry
import numpy as np
import project_utils as pu
from copy import deepcopy
import tf


class ScanFilter:
    def __init__(self):
        self.range_angle = {}
        rospy.init_node("scan_filter", anonymous=True)
        self.map_inflation_radius = float(rospy.get_param('~map_inflation_radius'))
        self.robot_count = rospy.get_param('~robot_count')
        self.robot_id = rospy.get_param('~robot_id')
        self.all_poses = {}
        self.robot_pose = (0, 0, 0)

        for i in range(self.robot_count):
            s = "def a_{0}(self, data): self.all_poses[{0}] = (data.pose.pose.position.x," \
                "data.pose.pose.position.y," \
                "(data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z," \
                "data.pose.pose.orientation.w), data.header.stamp.to_sec())".format(i)
            exec (s)
            exec ("setattr(ScanFilter, 'callback_pos_teammate{0}', a_{0})".format(i))
            exec ("rospy.Subscriber('/robot_{0}/base_pose_ground_truth', Odometry, self.callback_pos_teammate{0}, "
                  "queue_size = 100)".format(i))

        self.scan_subscriber = message_filters.Subscriber('base_scan', LaserScan)
        self.scan_pub = rospy.Publisher('filtered_scan', LaserScan, queue_size=1)
        self.base_pose_subscriber = message_filters.Subscriber('base_pose_ground_truth', Odometry)
        ats = ApproximateTimeSynchronizer([self.base_pose_subscriber, self.scan_subscriber], 10, 0.1)
        ats.registerCallback(self.topic_callback)

    def process_scan_message(self, msg, robot_ranges, robot_angles):
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max
        ranges = msg.ranges
        robot_count = len(robot_ranges)
        size = len(ranges)
        new_ranges = [0.0] * size
        for i in range(size):
            if range_min <= ranges[i] <= range_max:
                r = ranges[i]
                angle = angle_min + i * angle_increment
                is_robot = False
                for j in range(robot_count):
                    if (abs(robot_ranges[j] - r) < self.map_inflation_radius) and (
                            abs(robot_angles[j] - angle) < self.map_inflation_radius):
                        is_robot = True
                        break
                if is_robot:
                    new_ranges[i] = 0.0
                else:
                    new_ranges[i] = r
            else:
                new_ranges[i] = ranges[i]

        scan = LaserScan()
        scan.header.frame_id = msg.header.frame_id
        scan.header.stamp = msg.header.stamp
        scan.angle_min = msg.angle_min
        scan.angle_max = msg.angle_max
        scan.angle_increment = msg.angle_increment
        scan.time_increment = msg.time_increment
        scan.scan_time = msg.scan_time
        scan.range_min = msg.range_min
        scan.range_max = msg.range_max
        scan.ranges = new_ranges
        scan.intensities = msg.intensities
        self.scan_pub.publish(scan)

    def topic_callback(self, pose_msg, scan_msg):
        if len(self.all_poses) == self.robot_count:
            robot_pose = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, (
                pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w))
            scan_time = scan_msg.scan_time  # time (sec) between scans
            msg_time = scan_msg.header.stamp.to_sec()
            robot_ranges, robot_angles = self.find_other_robots(robot_pose, msg_time, scan_time)
            self.process_scan_message(scan_msg, robot_ranges, robot_angles)

    def find_other_robots(self, robot_pose, msg_time, scan_time):
        robot_ranges = []
        robot_angles = []
        rTm, mTr = self.generate_transformation_matrices(robot_pose)
        yaw = self.get_bearing(robot_pose[2])
        pose_arr = np.asarray([robot_pose[0], robot_pose[1], yaw, 1])
        robot_ids = list(self.all_poses)
        for rid in robot_ids:
            t = self.all_poses[rid][3]
            if abs(t - msg_time) <= 0.15:  # TODO value that can be inferred.
                other_pose = self.all_poses[rid]
                oTm, mTo = self.generate_transformation_matrices(other_pose)
                oTr = oTm.dot(mTr)
                other_robot_relative_pose = oTr.dot(pose_arr)
                distance = pu.D(robot_pose, other_robot_relative_pose)
                angle = pu.theta(robot_pose, other_robot_relative_pose)
                robot_ranges.append(distance)
                robot_angles.append(angle)
            else:
                rospy.logerr("difference {} {}".format(t - msg_time, scan_time))
        return robot_ranges, robot_angles

    def generate_transformation_matrices(self, pose):
        map_origin = (0, 0, 0, 1)
        theta = pu.theta(map_origin, pose)
        M1 = [[np.cos(theta), -1 * np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        T = deepcopy(M1)
        T.append([0, 0, 0, 1])
        yaw = self.get_bearing(pose[2])
        T[0].append(pose[0])
        T[1].append(pose[1])
        T[2].append(yaw)
        tT = np.asarray(T)
        tT_inv = np.linalg.inv(tT)
        return tT, tT_inv

    def get_bearing(self, quaternion):
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]
        return yaw


if __name__ == '__main__':
    filter = ScanFilter()
    rospy.spin()
