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

nrobots_all =[2] # [2, 4, 6]
methods = ["gvgexploration","recurrent_connectivity", "continuous_connectivity"]
runs = range(5)
envs =  {"office": [4.0, 10.0]} #{"office": [4.0, 10.0], "cave": [4.0, 10.0], "city": [4.0, 10.0]}
catkinws_path='/home/masaba/stage_ws'
package_path='{}/src/gvgexploration'.format(catkinws_path)
karto_pkg='gvgexplore'
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
                launcher_args = ['roslaunch', package, 'multirobot_{}.launch'.format(nrobots), "world:={}{}".format(world,nrobots) , "robot_count:={}".format(nrobots),"run:={}".format(run),"method:={}".format(package),"environment:={}".format(world),"bs_pose:={},{}".format(bs_pose[0],bs_pose[1])]
                rospy.logerr(launcher_args)
                main_process = subprocess.Popen(launcher_args)
                main_process.wait()
                check_kill_process("ros")
                # break
            break
        break
    break

