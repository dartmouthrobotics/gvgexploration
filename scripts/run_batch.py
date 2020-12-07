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

nrobots_all =[2]#, 6]

methods = ["recurrent_connectivity"] #["continuous_connectivity"]
runs = [3,4,5]
envs = {"office": [33.0, 20.0],"cave": [20.0, 8.0],"city": [25.0, 4.0]}
#envs = {"office": [33.0, 20.0], "cave": [20.0, 8.0], "city": [25.0, 4.0]}
#envs = {"city": [25.0, 4.0]}
#envs = {"cave": [20.0, 8.0]}
target_ratios=[0.05, 0.5]
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
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            print("No process")

def start_simulation(launcher_args):
    errors_before = num_errors()
    rospy.logerr(launcher_args)
    main_process = subprocess.Popen(launcher_args)
    main_process.wait()
    check_kill_process("ros")
    time.sleep(10)


for run in runs:
    for nrobots in nrobots_all:
        for package in methods:
            for world, bs_pose in envs.items():
                launcher_args = ['roslaunch', package, 'multirobot_{}.launch'.format(nrobots), "world:={}{}".format(world,nrobots) , "robot_count:={}".format(nrobots),"run:={}".format(run),"method:={}".format(package),"environment:={}".format(world),"bs_pose:={},{}".format(bs_pose[0],bs_pose[1])]
                launcher_args.append("max_target_info_ratio:={}".format(target_ratios[0]))
                if package=='recurrent_connectivity':
                    for t in target_ratios:
                        if os.path.exists("/home/albertoq/.ros/" + package + "/coverage_" + world + "_" + str(nrobots) + "_" + str(run) + "_1_" + str(t) + ".pickle"):
                            continue
                        launcher_args[-1] = "max_target_info_ratio:={}".format(t)
                        start_simulation(launcher_args)
                else:
                    if os.path.exists("/home/albertoq/.ros/" + package + "/coverage_" + world + "_" + str(nrobots) + "_" + str(run) + "_1_0.05.pickle"):
                        continue
                    start_simulation(launcher_args)
