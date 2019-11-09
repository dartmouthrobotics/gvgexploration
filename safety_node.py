#!/usr/bin/python
import numpy

import rospy
import tf

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from std_srvs.srv import Trigger

SAFE_DISTANCE = 0.7

def odometry_callback(odometry_msg):
    pose = odometry_msg.pose.pose
    quaternion = (
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w)
    euler = tf.transformations.euler_from_quaternion(quaternion)
    yaw = euler[2]
    rospy.loginfo("{} {} {}".format(pose.position.x, pose.position.y, yaw)) 

def laser_scan_callback(laser_scan_msg):
    ranges = numpy.array(laser_scan_msg.ranges)
    min_laser_scan_reading = ranges.min()
    reading_string = "Minimum laser scan reading: {}".format(min_laser_scan_reading)
    rospy.logdebug(reading_string) 

    # Assumption! Just stop once and exit.
    if min_laser_scan_reading < SAFE_DISTANCE:
        start_stop = rospy.ServiceProxy('start_stop', Trigger)
        response = start_stop()
        rospy.logdebug(response.message) 
        rospy.signal_shutdown(reading_string)

if __name__ == "__main__":
    rospy.init_node("safety") 
    rospy.Subscriber("odom", Odometry, odometry_callback)
    rospy.Subscriber("base_scan", LaserScan, laser_scan_callback)

    rospy.spin()
