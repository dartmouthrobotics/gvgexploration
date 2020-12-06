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
import rospkg

nrobots_all =[2] #[2, 4, 6]

methods = ["continuous_connectivity", "recurrent_connectivity"] #"gvgexploration","recurrent_connectivity",
runs = [0,1,2,3,4]
envs = {"office": [33.0, 20.0],"cave": [20.0, 8.0],"city": [25.0, 4.0]}
target_ratios=[0.05,0.5]
rospack = rospkg.RosPack()
package_path = rospack.get_path('gvgexploration') + "/log/errors.log" 

def num_errors():
    f = open(package_path, "r")
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

def start_simulation(launcher_args):
    errors_before = num_errors()
    rospy.logerr(launcher_args)
    main_process = subprocess.Popen(launcher_args)
    main_process.wait()
    check_kill_process("ros")
    time.sleep(10)


for nrobots in nrobots_all:
    for package in methods:
        i = 0
        for run in runs:
            for world, bs_pose in envs.items():
                launcher_args = ['roslaunch', package, 'multirobot_{}.launch'.format(nrobots), "world:={}{}".format(world,nrobots) , "robot_count:={}".format(nrobots),"run:={}".format(run),"method:={}".format(package),"environment:={}".format(world),"bs_pose:={},{}".format(bs_pose[0],bs_pose[1])]
                launcher_args.append("max_target_info_ratio:={}".format(target_ratios[i]))
                if package=='recurrent_connectivity':
                    for t in target_ratios:
                        launcher_args[-1] = "max_target_info_ratio:={}".format(target_ratios[i])
                        start_simulation(launcher_args)
                else:
                    start_simulation(launcher_args)
