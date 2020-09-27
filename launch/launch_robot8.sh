#!/bin/bash

sudo pkill -f wifi_node >/dev/null &2>&1 &
sudo  pkill -f roscore  >/dev/null &2>&1 &
sleep 5

roscore >/dev/null &2>&1 &
sleep 5
sudo airmon-ng start wlan1 >/dev/null &2>&1 &
sudo su -c ../../wifi_node/./launch_script.sh >/dev/null &2>&1 &

sudo sh -c "echo 1 >/proc/sys/net/ipv4/ip_forward"
sudo sh -c "echo 0 >/proc/sys/net/ipv4/icmp_echo_ignore_broadcasts"

roslaunch gvgexploration husarion.launch robot_id:=8 base_station:=1,5 robot_type:=2
