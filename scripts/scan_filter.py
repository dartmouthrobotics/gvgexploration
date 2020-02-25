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

    def process_scan_message(self, msg, robot_poses):#robot_ranges, robot_angles): # TODO cleanup the code.
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        range_min = msg.range_min
        range_max = msg.range_max
        ranges = msg.ranges
        robot_count = len(robot_poses)#ranges)
        size = len(ranges)
        new_ranges = [0.0] * size
        for i in range(size):
            if range_min <= ranges[i] <= range_max:
                r = ranges[i]
                angle = angle_min + i * angle_increment
                is_robot = False
                r_x, r_y = r * np.cos(angle), r * np.sin(angle)
                for j in range(robot_count):
                    """
                    if (abs(robot_ranges[j] - r) < self.map_inflation_radius) and (
                            abs(robot_angles[j] - angle) < 0.6): # TODO angle based on distance.
                    """
                    if abs(r_x - robot_poses[j][0]) < self.map_inflation_radius and abs(r_y - robot_poses[j][1]):
                        is_robot = True
                        break
                if is_robot:
                    new_ranges[i] = -1.0
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
            robot_poses = self.find_other_robots(robot_pose, msg_time, scan_time) #robot_ranges, robot_angles
            self.process_scan_message(scan_msg, robot_poses) #, robot_ranges, robot_angles)

    def find_other_robots(self, robot_pose, msg_time, scan_time):
        robot_ranges = []
        robot_angles = []
        robot_poses = []
        mTr, rTm = self.generate_transformation_matrices(robot_pose)
        robot_ids = list(self.all_poses)
        for rid in robot_ids:
            t = self.all_poses[rid][3]
            if abs(t - msg_time) <= 0.15:  # TODO value that can be inferred.
                other_pose = self.all_poses[rid]
                mTo, oTm = self.generate_transformation_matrices(other_pose)
                rTo = rTm.dot(mTo)
                distance = np.linalg.norm(rTo[0:3,3])
                angle = tf.transformations.euler_from_matrix(rTo[0:3,0:3])[2]
                robot_ranges.append(distance)
                robot_angles.append(angle)
                robot_poses.append((rTo[0,3], rTo[1,3]))
            else:
                rospy.logerr("difference {} {}".format(t - msg_time, scan_time))
        return robot_poses#robot_ranges, robot_angles

    def generate_transformation_matrices(self, pose):
        mTr = tf.transformations.quaternion_matrix(pose[2])
        mTr[0,3] = pose[0]
        mTr[1,3] = pose[1]
        mTr_inv = np.linalg.inv(mTr)
        return mTr, mTr_inv


if __name__ == '__main__':
    filter = ScanFilter()
    rospy.spin()
