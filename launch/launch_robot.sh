#!/bin/bash
nohup sudo airmon-ng start wlan1 &
sudo su -c ./launch_script.sh root
nohup roslaunch gvgexploration husarion.launch robot_id:=4 base_station:=5,7,8 robot_type:=2 &
