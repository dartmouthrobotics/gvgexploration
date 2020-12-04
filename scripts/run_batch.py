import os
import signal
import socket
import subprocess
import sys
import time
import sys
import numpy
import rospy
import rosgraph

nrobots_all =[6] #[2, 4, 6]
methods = ["gvgexploration"]
runs = [0,1,2,3,4]
envs = {"office": [33.0, 20.0]} #,"cave": [20.0, 8.0],
target_ratios=[0.05,0.5]
catkinws_path='/home/masaba/stage_ws'
package_path='{}/src/gvgexploration'.format(catkinws_path)
def num_errors():
    f = open("{}/log/errors.log".format(package_path), "r")
    lines = f.readlines()
    f.close()
    num = len(lines)
    return num

def check_kill_process(pstring):
    for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        print(line)
        os.kill(int(pid), signal.SIGKILL)

for nrobots in nrobots_all:
    for package in methods:
        for run in runs:
            for world, bs_pose in envs.items():
                errors_before = num_errors()
                launcher_args = ['roslaunch', package, 'multirobot_{}.launch'.format(nrobots), "world:={}{}".format(world,nrobots) , "robot_count:={}".format(nrobots),"run:={}".format(run),"method:={}".format(package),"environment:={}".format(world),"bs_pose:={},{}".format(bs_pose[0],bs_pose[1]),"max_target_info_ratio:={}".format(0.05)]
                rospy.logerr(launcher_args)
                main_process = subprocess.Popen(launcher_args)
                main_process.wait()
                check_kill_process("ros")
                time.sleep(10)
                # if package=='recurrent_connectivity':
                #     launcher_args = ['roslaunch', package, 'multirobot_{}.launch'.format(nrobots), "world:={}{}".format(world,nrobots) , "robot_count:={}".format(nrobots),"run:={}".format(run),"method:={}".format(package),"environment:={}".format(world),"bs_pose:={},{}".format(bs_pose[0],bs_pose[1]),"max_target_info_ratio:={}".format(0.5)]
                #     rospy.logerr(launcher_args)
                #     main_process = subprocess.Popen(launcher_args)
                #     main_process.wait()
                #     check_kill_process("ros")

                # break
        # break
    # break

