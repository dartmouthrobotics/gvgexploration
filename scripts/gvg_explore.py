#!/usr/bin/python

# Python modules
import numpy as np
import time  # clock.

# ROS related python modules
import rospy

import tf  # for TransformListener

import actionlib
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger, TriggerResponse
from nav2d_navigator.msg import MoveToPosition2DAction, MoveToPosition2DGoal
from std_msgs.msg import String
# Custom modules
from graph import Graph
import project_utils as pu
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, save_data
# TODO specify what is actually imported.
from gvgexploration.msg import *
from gvgexploration.srv import *


class GVGExplore:
    # States of the node
    IDLE = 0  # Not initiated.
    DECISION = 1  # Finding the next GVG
    MOVE_TO = 2  # Go to the next location
    MOVE_TO_LEAF = 3  # Initial motion before starting the DFS.

    def __init__(self):
        # GVG graph.
        self.graph = Graph()

        # List of previous poses.
        self.prev_pose = []

        # Current state of the node.
        self.current_state = self.IDLE

        # Parameters.
        self.robot_id = rospy.get_param('~robot_id')

        # Goal parameters.
        self.target_distance = rospy.get_param('/target_distance')
        self.target_angle = rospy.get_param('/target_angle')
        self.debug_mode = rospy.get_param('/debug_mode')
        self.method = rospy.get_param('/method')
        self.termination_metric = rospy.get_param("/termination_metric")
        self.environment=rospy.get_param("/environment")
        self.robot_count=rospy.get_param("/robot_count")
        self.max_target_info_ratio=rospy.get_param("/max_target_info_ratio")
        self.run=rospy.get_param("/run")
        self.current_pose=None
        self.previous_point=[]
        # nav2d MoveTo action.
        self.client_motion = actionlib.SimpleActionClient("/robot_{}/MoveTo".format(self.robot_id),
                                                          MoveToPosition2DAction)
        self.client_motion.wait_for_server()

        # Service to start or stop the gvg exploration.
        self.start_gvg_explore = rospy.Service('/robot_{}/gvg/start_stop'.format(self.robot_id), Trigger,
                                               self.start_stop)



        # tf listener.
        self.listener = tf.TransformListener()
        self.explore_computation = []
        self.traveled_distance=[]


        rospy.Subscriber('/shutdown', String, self.save_all_data)
        self.intersec_pub = rospy.Publisher('intersection', Pose, queue_size=0)
        # Topics for robot.py # TODO restructure it.
        rospy.Subscriber('/robot_{}/gvgexplore/goal'.format(self.robot_id), Pose, self.initial_action_handler)
        rospy.Service('/robot_{}/gvgexplore/cancel'.format(self.robot_id), CancelExploration,
                      self.received_prempt_handler)
        self.goal_feedback_pub = rospy.Publisher("/robot_{}/gvgexplore/feedback".format(self.robot_id), Pose,
                                                 queue_size=1)

        rospy.loginfo("Robot {}: Exploration server online...".format(self.robot_id))

    # def start_stop(self, req):
    #     if self.current_state == self.IDLE:
    #         self.current_state = self.INITIAL_COMMUNICATE
    #     else:
    #         self.client_motion.cancel_goal()
    #         self.current_state = self.IDLE
    #     return TriggerResponse()

    def start_stop(self, req):
        if self.current_state != self.IDLE:
            self.client_motion.cancel_goal()
            self.current_state = self.IDLE
        return TriggerResponse()

    def move_robot_to_goal(self, goal, theta=0):
        if len(self.previous_point)==0:
            self.previous_point= self.get_robot_pose()
        pu.log_msg(self.robot_id,"Prev: {}, Current: {}".format(self.previous_point,goal),1)
        self.traveled_distance.append({'time': rospy.Time.now().to_sec(),'traved_distance': pu.D(self.previous_point, goal)})
        self.previous_point=goal

        self.current_point = goal
        move = MoveToPosition2DGoal()
        frame_id = 'robot_{}/map'.format(self.robot_id)  # TODO check.
        move.header.frame_id = frame_id
        move.target_pose.x = goal[INDEX_FOR_X]
        move.target_pose.y = goal[INDEX_FOR_Y]
        move.target_pose.theta = theta
        move.target_distance = self.target_distance
        move.target_angle = self.target_angle

        self.last_distance = -1
        self.client_motion.send_goal(move, feedback_cb=self.feedback_motion_cb)
        self.prev_pose.append(self.current_pose)
        if self.client_motion.wait_for_result():
            state = self.client_motion.get_state()
            if self.current_state == self.MOVE_TO:
                if state == GoalStatus.SUCCEEDED:
                    self.prev_pose += self.path_to_leaf[1:]
                    self.current_pose = goal
                else:
                    self.current_pose = self.get_robot_pose()
                    self.prev_pose += self.path_to_leaf[1:pu.get_closest_point(self.current_pose, np.array(self.path_to_leaf))[0]+1]
                self.current_state = self.DECISION
            elif self.current_state == self.MOVE_TO_LEAF:
                self.current_state = self.DECISION
                self.current_pose = self.get_robot_pose()

                # publish to feedback
                pose = Pose()
                pose.position.x = self.current_pose[INDEX_FOR_X]
                pose.position.y = self.current_pose[INDEX_FOR_Y]
                self.goal_feedback_pub.publish(pose)
            # rospy.sleep(1)

    def feedback_motion_cb(self, feedback):
        if self.current_state == self.MOVE_TO_LEAF:
            self.prev_pose.append(self.get_robot_pose())

        if (np.isclose(self.last_distance, feedback.distance) or \
                np.isclose(feedback.distance, self.last_distance)):
            self.same_location_counter += 1
            if self.same_location_counter > 5:  # TODO parameter:
                self.client_motion.cancel_goal()
        else:
            self.last_distance = feedback.distance
            self.same_location_counter = 0

        cpose = self.get_robot_pose()
        if self.graph.should_communicate(cpose):
            # self.current_state = self.COMMUNICATE
            pu.log_msg(self.robot_id, "communicate", self.debug_mode)
            cp = Pose()
            cp.position.x = cpose[INDEX_FOR_X]
            cp.position.x = cpose[INDEX_FOR_X]
            self.intersec_pub.publish(cp)

        if rospy.Time.now() - self.graph.latest_map.header.stamp > rospy.Duration(10):  # TODO parameter:
            self.graph.generate_graph()
            if not self.graph.latest_map.is_frontier(
                    self.prev_goal_grid,
                    self.goal_grid, self.graph.min_range_radius):
                self.client_motion.cancel_goal()

    def get_robot_pose(self):
        robot_pose = None
        while not robot_pose:
            try:
                self.listener.waitForTransform("robot_{}/map".format(self.robot_id),
                                               "robot_{}/base_link".format(self.robot_id),
                                               rospy.Time(0),
                                               rospy.Duration(4.0))
                (robot_loc_val, rot) = self.listener.lookupTransform("robot_{}/map".format(self.robot_id),
                                                                     "robot_{}/base_link".format(self.robot_id),
                                                                     rospy.Time(0))
                robot_pose = robot_loc_val[0:2]
            except:
                rospy.sleep(1)
                pass
        robot_pose = np.array(robot_pose)
        return robot_pose

    def spin(self):
        r = rospy.Rate(1)  # TODO frequency as parameter
        self.current_pose = self.get_robot_pose()
        while not rospy.is_shutdown():
            if self.current_state == self.DECISION:
                self.graph.generate_graph()
                pu.log_msg(self.robot_id,"Graph generated",self.debug_mode)

                start_time_clock = time.clock()
                self.path_to_leaf = self.graph.get_successors(self.current_pose,
                                                              self.prev_pose)
                end_time_clock = time.clock()
                gvg_time=end_time_clock - start_time_clock
                pu.log_msg(self.robot_id,"next path time {}".format(gvg_time),1-self.debug_mode)
                self.explore_computation.append({'time': rospy.Time.now().to_sec(),'robot_id':self.robot_id,'gvg_compute': gvg_time})
                if self.path_to_leaf:
                    if len(self.path_to_leaf) > 1:
                        prev_pose = self.path_to_leaf[-2]
                    else:
                        prev_pose = self.current_pose
                    self.prev_goal_grid = self.graph.latest_map.pose_to_grid(prev_pose)
                    self.goal_grid = self.graph.latest_map.pose_to_grid(self.path_to_leaf[-1])

                    self.current_state = self.MOVE_TO
                    self.move_robot_to_goal(self.path_to_leaf[-1], pu.angle_pq_line(self.path_to_leaf[-1], prev_pose))
                else:
                    pu.log_msg(self.robot_id,"no more leaves",self.debug_mode)
                    self.current_state = self.IDLE

            r.sleep()

    def save_all_data(self, data):
        """ Save data and kill yourself"""
        pu.save_data(self.explore_computation, '{}/explore_computation_{}_{}_{}_{}_{}_{}.pickle'.format(self.method, self.environment,self.robot_count,self.run,self.termination_metric,self.robot_id,self.max_target_info_ratio))
        pu.save_data(self.traveled_distance,'{}/traveled_distance_{}_{}_{}_{}_{}_{}.pickle'.format(self.method, self.environment,self.robot_count,self.run,self.termination_metric,self.robot_id,self.max_target_info_ratio))
        rospy.signal_shutdown("Shutting down GVG explore")

    def initial_action_handler(self, leaf):
        pu.log_msg(self.robot_id,"GVGExplore received new goal",self.debug_mode)
        self.graph.generate_graph()
        # if not self.current_pose:
        #     self.current_pose = self.get_robot_pose()

        self.prev_goal_grid = self.graph.latest_map.pose_to_grid(self.current_pose)
        self.goal_grid = self.graph.latest_map.pose_to_grid(np.array([leaf.position.x, leaf.position.y]))
        self.current_state = self.MOVE_TO_LEAF
        self.move_robot_to_goal(np.array([leaf.position.x, leaf.position.y]))

    def received_prempt_handler(self, data):
        pu.log_msg(self.robot_id,"GVGExplore action preempted",self.debug_mode)
        self.client_motion.cancel_goal()
        self.current_state = self.IDLE
        return CancelExplorationResponse(result=1)


if __name__ == "__main__":
    rospy.init_node("gvg_explore_node")

    gvg_explore = GVGExplore()
    gvg_explore.spin()
