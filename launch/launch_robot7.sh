#!/bin/bash
sudo sh -c "echo 1 >/proc/sys/net/ipv4/ip_forward"
sudo sh -c "echo 0 >/proc/sys/net/ipv4/icmp_echo_ignore_broadcasts"

roslaunch gvgexploration husarion.launch robot_id:=7 base_station:=4,8,5 robot_type:=2
