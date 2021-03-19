#!/usr/bin/python
import numpy as np

import rospy
import tf
from message_filters import TimeSynchronizer
import message_filters
from scipy.spatial import Voronoi, voronoi_plot_2d
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,Pose,PoseStamped
import project_utils as pu
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class GVGBuilder:
    """Filtering of the laser scans measurements corresponding to other robots.
    """
    def __init__(self):
        """Constructor."""
        # Node initialization.
        rospy.init_node("gvg_builder", anonymous=True)
        rospy.sleep(1)

        # Parameter reading.
        self.map_inflation_radius = float(rospy.get_param('map_inflation_radius'))
        self.robot_count = rospy.get_param('/robot_count')

        # Robot ID starting from 1, as navigation_2d
        self.robot_id = rospy.get_param('~robot_id')
        self.grid_resolution = rospy.get_param('/robot_{}/Mapper/grid_resolution'.format(self.robot_id))
        # Assumes that apart from the readings, and timestamp, the rest won't change.
        self.scan = rospy.wait_for_message('base_scan', LaserScan)
        self.scan.ranges = list(self.scan.ranges)
        self.scan_ranges_size = len(self.scan.ranges)
        self.initial_robot_pose=None
        self.initial_map_pose=None

        # Subscriber of the own robot.
        self.base_pose_subscriber = message_filters.Subscriber('base_pose_ground_truth', Odometry)
        self.scan_subscriber = message_filters.Subscriber('filtered_scan', LaserScan)
        self.obstacle_pose_pub=rospy.Publisher('laser_obstacles', Marker, queue_size=0)
        self.listener = tf.TransformListener()
        ats = TimeSynchronizer([self.base_pose_subscriber, self.scan_subscriber], 1)#, 0.01)
        ats.registerCallback(self.pose_scan_callback)
        rospy.Subscriber('/shutdown', String, self.save_all_data)



    def pose_scan_callback(self, pose_msg, scan_msg):
        """publishing poses of the scans.

        Args:
            robot_pose (x,y,quaternion)
            scan_msg (LaserScan)
        """
        robot_pose = (pose_msg.pose.pose.position.x, pose_msg.pose.pose.position.y, (pose_msg.pose.pose.orientation.x, pose_msg.pose.pose.orientation.y, pose_msg.pose.pose.orientation.z,pose_msg.pose.pose.orientation.w))
        scan_msg_stamp = scan_msg.header.stamp.to_sec()
        scan_time = scan_msg.scan_time
        ranges = scan_msg.ranges
        n=len(ranges)
        (roll, pitch, yaw) = euler_from_quaternion (robot_pose[2])
        angles = np.array([scan_msg.angle_min+yaw]*n)+np.arange(n)*scan_msg.angle_increment
        pose_vals = np.array([robot_pose[0],robot_pose[1],robot_pose[2][0], robot_pose[2][1], robot_pose[2][2],robot_pose[2][3]])
        robot_poses=np.repeat(pose_vals,[n]*len(pose_vals),axis=0).reshape(-1,n).T
        obstacle_poses = np.concatenate((np.multiply(ranges,np.cos(angles)).reshape(-1,1),np.multiply(ranges,np.sin(angles)).reshape(-1,1),np.zeros((n,4))), axis=1)
        scan_positions=robot_poses+obstacle_poses
        scan_poses=np.array([[scan_positions[i,0],scan_positions[i,1]] for i in range(n) if scan_msg.range_min<=ranges[i]<scan_msg.range_max])
        self.generate_gvg(robot_pose,scan_poses)
        self.publish_obstacles(scan_poses)



    def generate_gvg(self,rpose,scan_poses):
        pass

        # N=scan_poses.shape[0]
        # distances=-1*np.ones(N)
        # for i in range(N):
        #     distances[i]=round(pu.T(rpose,scan_poses[i,:]),6)
        # pose_tally=np.asarray(np.unique(distances, return_counts=True)).T
        # equidist_2 = pose_tally[pose_tally[:,1] ==2]
        # equidist_more = pose_tally[pose_tally[:,1] >2]
        # idx2_pairs=[]
        # idx_more_pairs=[]
        # if equidist_2.shape[0]:
        #     scan_poses[np.where(pose_tally == equidist_2[i,0]),:]
        #     idx2_pairs=[[rpose[0]+] for i in range(equidist_2.shape[0]) if equidist_2[i,0] !=-1]

        # if equidist_2.shape[0]:
        #     idx2_pairs=[scan_poses[np.where(pose_tally == equidist_2[i,0]),:] for i in range(equidist_2.shape[0]) if equidist_2[i,0] !=-1]
        # if equidist_more.shape[0]:
        #     idx_more_pairs=[scan_poses[np.where(pose_tally == equidist_more[i,0]),:] for i in range(equidist_more.shape[0]) if equidist_more[i,0] !=-1]




    def publish_obstacles(self,scan_poses):
        # Publish visited vertices
        m = Marker()
        m.id = 2
        m.header.frame_id = "robot_{}/map".format(self.robot_id)
        m.type = Marker.POINTS
        # TODO constant values set at the top.
        m.color.a = 1.0
        m.color.g = 0.5
        if self.robot_id==0:
            m.color.b = 0.5
        elif self.robot_id==1:
            m.color.g = 0.5
        m.scale.x = 0.2
        m.scale.y = 0.2
        for pose in scan_poses:
            p_ros = Point(x=pose[0], y=pose[1])
            m.points.append(p_ros)
        self.obstacle_pose_pub.publish(m)


    def publish_edge_areas(self,scan_poses):
        # Publish visited vertices
        m = Marker()
        m.id = 2
        m.header.frame_id = "robot_{}/map".format(self.robot_id)
        m.type = Marker.POINTS
        # TODO constant values set at the top.
        m.color.a = 1.0
        m.color.g = 0.5
        if self.robot_id==0:
            m.color.b = 0.5
        elif self.robot_id==1:
            m.color.g = 0.5
        m.scale.x = 0.2
        m.scale.y = 0.2
        for pose in scan_poses:
            p_ros = Point(x=pose[0], y=pose[1])
            m.points.append(p_ros)
        self.obstacle_pose_pub.publish(m)


    def save_all_data(self,data):
        rospy.signal_shutdown("Shutting down Scan filter")
if __name__ == '__main__':
    gvg_builder = GVGBuilder()
    rospy.spin()
