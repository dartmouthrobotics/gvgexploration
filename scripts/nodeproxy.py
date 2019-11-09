#!/usr/bin/env python
import time

import rospy
from std_msgs.msg import *
from std_srvs.srv import *
from nav2d_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
import json
import copy
import math
from threading import Lock
import numpy as np

BS_TYPE = 1
RR_TYPE = 2
FR_TYPE = 3

WIFI_RANGE = 100
BLUETOOTH_RANGE = 10

class nodeproxy(object):
    def __init__(self):

        self.wavelength = 0.12  # 12 centimeters for 2.4GHz WIFI
        self.communication_model = 0
        self.check_bandwidth = 0  # check band width before publishing a message
        self.constant_distance_model = self.compute_constant_distance_ss()
        self.msg_size = 976  # TODO automatically calculate the size of the message.

        self.ss = {}
        self.bandwidth = {}
        self.distances = {}
        self.tp = {}
        self.publish_time = {}
        self.connections = {}

        self.lock = Lock()
        rospy.init_node("nodeproxy", anonymous=True)

        self.robots_map = {}
        self.robot_type_map = {}
        robot_map = str(rospy.get_param("~robot_map", ''))
        self.parse_robot_map(robot_map)

        self.lasttime_before_performance_calc =rospy.Time.now().to_sec()
        self.robot_pose = {}

        self.message_queue={}

        rospy.Subscriber("/karto_out", LocalizedScan, self.robots_karto_out_callback)
        self.num_robots = 17
        self.publishers = {}
        for i in range(self.num_robots):
            self.publishers[i]=rospy.Publisher("/commsim/robot_{}/karto_in".format(i), LocalizedScan, queue_size=10)

        for i in range(self.num_robots):
            # Initialization of some data structures.
            self.tp[i] = self.msg_size

            # Callback code generation.
            s = "def a_{}(self, data): self.robot_pose[{}] = data.pose.pose.position".format(i,i)
            exec(s)
            exec("setattr(nodeproxy, 'callback_pos_teammate{}', a_{})".format(i,i))
            exec("rospy.Subscriber('/robot_{}/base_pose_ground_truth', Odometry, self.callback_pos_teammate{}, "
                 "queue_size = 100)".format(i,i))

        print("Node Proxy: Inititiazation Done...")

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            try:
                # self.compute_performance()
                r.sleep()
                # time.sleep(10)
            except Exception as e:
                print('interrupted!')
                rospy.loginfo(e.message)


    def parse_robot_map(self, robot_map):
        # robot - id - robot_type:r1, r2, rx;robot - id - robot_type:r1, r2, rx
        robot_map = "0-1:1,2,3,4;1-2:5,6,7;2-2:8,9,10;3-2:11,12,13;4-2:14,15,16;5-3:1;6-3:1;7-3:1;8-3:2;9-3:2;10-3:2;11-3:3;12-3:3;13-3:3;14-3:4;15-3:4;16-3:4"
        map_tokens= robot_map.split(';')
        for token in map_tokens:
            robot_tokens=token.split(':')
            robot_id_type_tokens=robot_tokens[0].split('-')

            robot_id = int(robot_id_type_tokens[0])
            robot_type = int(robot_id_type_tokens[1])
            robots = robot_tokens[1].split(',')
            robots=[int(r) for r in robots]
            self.robots_map[robot_id]=robots
            self.robot_type_map[robot_id]=robot_type

        print( self.robots_map)
    def robots_karto_out_callback(self, data):
        robot_id= data.robot_id-1
        recipients=self.robots_map[robot_id]

        for i in recipients:
            if self.can_communicate(robot_id, i, model=self.communication_model):
                # publish old messages
                self.publish_old_messages(robot_id,i)
                # publish new message
                self.publishers[i].publish(data)
                with open('nav_performance{}_{}.txt'.format(self.num_robots, self.communication_model), 'a+') as fd:
                    fd.write("passed=={}-{}\n".format(robot_id,i))
            else:
                if robot_id in self.message_queue:
                    self.message_queue[robot_id].append(data)
                else:
                    self.message_queue[robot_id]=[data]
                rospy.loginfo("{}-{}".format(robot_id,i))
                # save message for sending later

    def publish_old_messages(self,sender_id,receiver_id):
        if sender_id in self.message_queue:
            pending_messages=self.message_queue[sender_id]
            msg_copy= copy.deepcopy(pending_messages)
            for i in range(len(msg_copy)):
                msg= msg_copy[i]
                self.publishers[receiver_id].publish(msg)
                self.message_queue[sender_id].remove(msg)
        pass
    def compute_performance(self):
        self.lock.acquire()
        try:
            current_time = rospy.Time.now().to_sec()

            data = {'start_time':  current_time}
            throughput={}
            for i in range(0, self.num_robots):
                cp = {}
                overhead={}

                if i in self.connections:
                    cp = {t: v for t, v in self.connections[i].items() if self.lasttime_before_performance_calc < t <= current_time}
                throughput['commsim_{}'.format(i)]= cp

                if i in self.publish_time:
                    overhead = {t: v for t, v in self.publish_time[i].items() if self.lasttime_before_performance_calc < t <= current_time}

                data['overhead_{}'.format(i)] = overhead

                for j in range(0,self.num_robots):
                    if i !=j:
                        combn=(i,j)
                        cp_ij={}
                        distance={}
                        ss={}
                        bw={}
                        if combn in self.connections:
                            cp_ij = {t: v for t, v in self.connections[combn].items() if self.lasttime_before_performance_calc < t <= current_time}
                        if combn in self.distances:
                            distance = {t: v for t, v in self.distances[combn].items() if self.lasttime_before_performance_calc < t <= current_time}
                        if combn in self.ss:
                            ss = {t: v for t, v in self.ss[combn].items() if self.lasttime_before_performance_calc < t <= current_time}
                        if combn in self.bandwidth:
                            bw = {t: v for t, v in self.bandwidth[combn].items() if self.lasttime_before_performance_calc < t <= current_time}

                        data['distance_{}'.format(combn)] = distance
                        data['signal_strength_{}'.format(combn)] =ss
                        data['bw_{}'.format(combn)] = bw
                        throughput['commsim_{}'.format(combn)]=cp_ij
            data['throughput'] = throughput
            json_data = json.dumps(data)
            json_str = str(json_data)
            # self.commsim_performance.publish(json_str)
            with open('nav_performance{}_{}.txt'.format(self.num_robots,self.communication_model), 'a+') as fd:
                fd.write("{}\n".format(json_str))
            self.lasttime_before_performance_calc = current_time

        except Exception as e:
            print("getting error: {}".format(e.message))
        finally:
            self.lock.release()

    def compute_distance(self, loc1, loc2):
        distance = math.sqrt(((loc1.x - loc2.x) ** 2) + ((loc1.y - loc2.y) ** 2))
        with open('nav_performance{}_{}.txt'.format(self.num_robots, self.communication_model), 'a+') as fd:
            fd.write("distance=={}-{}-{}\n".format(loc1, loc2, distance))
        return distance

    def can_communicate(self, robot1, robot2, model=1):
        # check if pose of both robots is known
        if robot1 in self.robot_pose and robot2 in self.robot_pose:
            # compute the distance between them
            distance = self.compute_distance(self.robot_pose[robot1], self.robot_pose[robot2])
            # check the robot type of  the receiver and if the sender is one of the receiver's assigned robots
            if self.robot_type_map[robot1]== BS_TYPE and robot2 in self.robots_map[robot1]:
                # checking if they're within communication range
                if distance <=WIFI_RANGE:
                    return True
            elif self.robot_type_map[robot1]== RR_TYPE and robot2 in self.robots_map[robot1]:
                if self.robot_type_map[robot2]==BS_TYPE and distance<=WIFI_RANGE:
                    return True
                elif self.robot_type_map[robot2]==FR_TYPE and distance<=BLUETOOTH_RANGE:
                    return True
            elif self.robot_type_map[robot1] == FR_TYPE and robot2 in self.robots_map[robot1]:
                if distance<=BLUETOOTH_RANGE:
                    return True

        else:
            print(robot1,robot2)
            
        return False

    def compute_percentage_bandwidth(self, robot1, robot2, current_time):
        bandwidth_percent = 0
        bandwidth_value = 0
        combn=(robot1,robot2)
        max_bandwidth = np.nansum([self.tp[i] for i in range(self.num_robots) if i in self.tp])

        # if robot1 in self.robot_pose and robot2 in self.robot_pose:
        bw_12 = 0
        counter = 0
        if current_time in self.tp[combn]:
            bw_12 += self.tp[combn][current_time]
            counter += 1
        if current_time in self.tp[combn]:
            bw_12 += self.tp[combn][current_time]
            counter += 1
        if counter > 0:  # == 0 shouldn't happen, just for safety. TODO remove.
            bw_12 = bw_12 / float(counter)
        if max_bandwidth>0:
            bandwidth_percent = round((bw_12 / max_bandwidth) * 100, 2)
        bandwidth_value = bw_12
        return bandwidth_percent, bandwidth_value

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


if __name__ == '__main__':
    node = nodeproxy()
    node.spin()