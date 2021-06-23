#!/usr/bin/python
import numpy as np
import rospy
import tf
from message_filters import TimeSynchronizer
import message_filters
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class ScanFilter:
    """Filtering of the laser scans measurements corresponding to other robots.
    """

    def __init__(self):
        """Constructor."""
        # Node initialization.
        rospy.init_node("scan_filter", anonymous=True)
        rospy.sleep(1)

        # Parameter reading.
        self.map_inflation_radius = float(rospy.get_param('map_inflation_radius'))
        self.robot_count = rospy.get_param('/robot_count')
        # Robot ID starting from 1, as navigation_2d
        self.robot_id = rospy.get_param('~robot_id')
        rospy.logerr("Received Robot ID: {} Robot Count: {}".format(self.robot_id, self.robot_count))
        # Initialization of variables.
        self.all_poses = {}  # Poses of other robots in the world reference frame.

        # Get pose from other robots.        
        for i in xrange(0, self.robot_count):
            if i != self.robot_id:
                s = "def a_{0}(self, data): self.all_poses[{0}] = (data.pose.pose.position.x," \
                    "data.pose.pose.position.y," \
                    "(data.pose.pose.orientation.x,data.pose.pose.orientation.y,data.pose.pose.orientation.z," \
                    "data.pose.pose.orientation.w), data.header.stamp.to_sec())".format(i)
                exec(s)
                exec("setattr(ScanFilter, 'callback_pos_teammate{0}', a_{0})".format(i))
                exec("rospy.Subscriber('/robot_{0}/base_pose_ground_truth', Odometry, self.callback_pos_teammate{0}, "
                     "queue_size=1)".format(i))

        # Assumes that apart from the readings, and timestamp, the rest won't change.
        self.scan = rospy.wait_for_message('base_scan', LaserScan)
        self.scan.ranges = list(self.scan.ranges)
        self.scan_ranges_size = len(self.scan.ranges)

        self.pose_pub = rospy.Publisher('/robot_{}/robot_pose'.format(self.robot_id), Marker, queue_size=0)
        self.marker_colors = rospy.get_param('/robot_colors')
        self.marker_id = self.robot_id

        # Subscriber of the own robot.
        self.base_pose_subscriber = message_filters.Subscriber('base_pose_ground_truth', Odometry)
        self.scan_subscriber = message_filters.Subscriber('base_scan', LaserScan)
        ats = TimeSynchronizer([self.base_pose_subscriber, self.scan_subscriber], 1)  # , 0.01)
        ats.registerCallback(self.pose_scan_callback)

        # Publisher of the filtered scan.
        self.scan_pub = rospy.Publisher('filtered_scan', LaserScan, queue_size=1)
        # Creation of the message.

        rospy.Subscriber('/shutdown', String, self.save_all_data)

    def pose_scan_callback(self, pose_msg, scan_msg):
        """Callback to processing scan of the robot.

        Args:
            pose_msg (Odometry)
            scan_msg (LaserScan)
        """

        # If got the pose of other robots, then the scan message can be processed.
        if len(self.all_poses) == self.robot_count - 1:
            robot_pose = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, (
                pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z,
                pose_msg.pose.pose.orientation.w))

            self.publish_visited_vertices(self.robot_id, robot_pose)
            scan_msg_stamp = scan_msg.header.stamp.to_sec()
            scan_time = scan_msg.scan_time  # time (sec) between scans

            # Get other robot poses in the robot reference frame.
            robot_poses = self.find_other_robots(robot_pose, scan_msg_stamp, scan_time)

            # Filter the scan message.
            self.process_scan_message(scan_msg, robot_poses)

    def find_other_robots(self, robot_pose, scan_msg_stamp, scan_time):
        """Transform other robot poses in the robot reference frame.

        Args:
            robot_pose (list): [x, y, (qx, qy, qz, qw)]
            scan_msg_stamp (float): timestamp of the scan message.
            scan_time (float): time between scans.
        """

        robot_poses = {}  # x,y of other robots in the robot reference frame.

        # Transformation matrices from world to robot, and from robot to world.
        rTm = self.generate_transformation_matrices(robot_pose, inverse=True)

        # For each other robot.
        for rid in list(self.all_poses):
            t = self.all_poses[rid][3]  # timestamp of other robot pose.

            # If the difference between messages is small.
            if abs(t - scan_msg_stamp) <= scan_time + 0.15:
                # Transformation matrices from world to other robot and viceversa.
                mTo = self.generate_transformation_matrices(self.all_poses[rid], inverse=False)
                # Transformation matrix between robot and other robot.
                rTo = rTm.dot(mTo)
                distance = np.linalg.norm(rTo[0:3, 3])
                # angle = tf.transformations.euler_from_matrix(rTo[0:3,0:3])[2]
                if distance < self.scan.range_max:
                    # Add only if within sensor range.
                    robot_poses[rid] = [rTo[0, 3], rTo[1, 3]]
            else:
                rospy.logerr("difference {} {}".format(t - scan_msg_stamp, scan_time))

        return robot_poses

    def process_scan_message(self, scan_msg, robot_poses):
        """Filter the scan message.

        Args:
            scan_msg (LaserScan)
            robot_poses (dict): key-robot_id, value-[x,y]
        """
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        range_min = scan_msg.range_min
        range_max = scan_msg.range_max
        ranges = scan_msg.ranges
        new_ranges = self.scan.ranges
        # Check each range, whether is hitting a robot or not.
        for i in xrange(self.scan_ranges_size):
            r = ranges[i]

            # If valid range value.
            if range_min <= r <= range_max:
                is_robot = False  # Hit is another robot.
                angle = angle_min + i * angle_increment
                r_x, r_y = r * np.cos(angle), r * np.sin(angle)
                for robot_id in robot_poses.keys():
                    if abs(r_x - robot_poses[robot_id][0]) < self.map_inflation_radius \
                            and abs(r_y - robot_poses[robot_id][1]) < self.map_inflation_radius:
                        is_robot = True
                        break
                if is_robot:
                    # Value arbitrarily set for nav2d_karto.
                    new_ranges[i] = np.nan
                else:
                    new_ranges[i] = r
            else:
                new_ranges[i] = r

        self.scan.header.stamp = scan_msg.header.stamp
        self.scan.intensities = scan_msg.intensities
        self.scan_pub.publish(self.scan)

    def generate_transformation_matrices(self, pose, inverse):
        """Transformation matrices from pose.

        Args:
            pose (list): x, y, (qx, qy, qz, qw)
            inverse (bool): return the inverse.

        Return:
            Transformation matrices.
        """

        mTr = tf.transformations.quaternion_matrix(pose[2])

        mTr[0, 3] = pose[0]
        mTr[1, 3] = pose[1]

        if inverse:
            # Manual calculation of the inverse because of optimization.
            rRm = np.transpose(mTr[0:3, 0:3])
            rtm = -rRm.dot(mTr[0:3, 3])
            mTr[0:3, 0:3] = rRm
            mTr[0:3, 3] = rtm
            return mTr  # np.linalg.inv(mTr)
        else:
            return mTr

    def publish_visited_vertices(self, robot_id, pose):
        rid = str(robot_id)
        m = Marker()
        m.id = self.marker_id
        m.header.frame_id = 'robot_{}/map'.format(self.robot_id)
        m.type = Marker.POINTS
        # TODO constant values set at the top.
        m.color.a = 1.0
        m.color.b = self.marker_colors[rid][2]
        m.color.g = self.marker_colors[rid][1]
        m.color.r = self.marker_colors[rid][0]
        m.scale.x = 0.5
        m.scale.y = 0.5
        m.lifetime = rospy.rostime.Duration(0.01)
        p_ros = Point(x=pose[0], y=pose[1])
        m.points.append(p_ros)
        self.pose_pub.publish(m)
        self.marker_id += 1

    def save_all_data(self, data):
        rospy.signal_shutdown("Shutting down Scan filter")


if __name__ == '__main__':
    scan_filter = ScanFilter()
    rospy.spin()
