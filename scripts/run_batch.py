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

nrobots_all = [2, 4, 6]
methods = {"gvg": "gvgexplore"} #, "recurrent": "recurrent_connectivity", "continuous": "continuous_connectivity"
runs = range(5)
envs = {"office": [4.0, 10.0]}  # open  #, "cave": [4.0, 10.0], "city": [4.0, 10.0]
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
    for env, package in methods.items():
        for run in runs:
            for world, bs_pose in envs.items():
                errors_before = num_errors()

                launcher_args = ['roslaunch', package, 'multirobot_{}.launch'.format(nrobots), "world:={}_{}".format(world,nrobots) , "robot_count:={}".format(nrobots),"run:={}".format(run),"method:={}".format(env),"environment:={}".format(env),"bs_pose:={},{}".format(bs_pose[0],bs_pose[1])]
                rospy.logerr(launcher_args)
                main_process = subprocess.Popen(launcher_args)
                time.sleep(8)
                ros_started = False
                while not ros_started:
                    try:
                        rosgraph.Master('/rostopic').getPid()
                        rospy.logerr("LAUNCH COMPLETE!!")
                        ros_started = True
                    except socket.error:
                        raise rostopic.ROSTopicIOException("Unable to communicate with master!")

                time.sleep(5)
                karto_args = ['roslaunch', karto_pkg, 'karto_{}.launch'.format(nrobots), "robot_count:={}".format(nrobots) , "run:={}".format(run),"method:={}".format(env)]
                karto_process = subprocess.Popen(karto_args)
                time.sleep(5)
                localizer_command = ".{}/localizer/maze_localize".format(package_path) #".{}/localizer/{}_{}_localize".format(package_path,world, nrobots)
                localizer_args = [localizer_command]
                os.system("cd {}/localizer && ./maze_localize".format(package_path))
                os.system("./maze_localize")
                time.sleep(5)

                main_process.wait()
                errors_after = num_errors()
                print("finished")
                time.sleep(3)
                check_kill_process("ros")
