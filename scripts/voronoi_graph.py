#!/usr/bin/python
import random
import itertools
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial import Voronoi, voronoi_plot_2d
import rospy
from nav_msgs.msg import *
import os
import math
import pickle

import numpy as np
import scipy.misc

SCALE = 10  # TODO as param
FREE = 0
OCCUPIED = 100
UNKNOWN = -1

ROBOT_SPACE = 1.0  # TODO pick from param (robot radius + inflation radius)

OUT_OF_BOUNDS = -2


def get_neighboring_states(pose, latest_map):
    x = pose[0]
    y = pose[1]
    poses = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x - 1, y - 1)]
    states = [latest_map[p] for p in poses if p in latest_map]
    return states


def get_map_points(map_message):
    obstacles = []
    free_space = []
    unknown = []
    latest_map = {}
    map_neighborhood = {}
    map_info = map_message.info
    grid_values = map_message.data
    origin_pose = map_info.origin.position
    map_width = map_info.width
    map_height = map_info.height
    resolution = map_info.resolution

    origin_pose_x = origin_pose.x
    origin_pose_y = origin_pose.y
    for col in range(map_width):
        map_x = math.floor(origin_pose_x + col * resolution)
        for row in range(map_height):
            data_val = grid_values[row * map_width + col]
            map_y = math.floor(origin_pose_y + row * resolution)
            if data_val == OCCUPIED:
                obstacles.append((map_x, map_y))
            elif data_val == FREE:
                free_space.append((map_x, map_y))
            else:
                unknown.append((map_x, map_y))
            latest_map[(map_x, map_y)] = data_val

            neighbor_states = get_neighboring_states((map_x, map_y), latest_map)
            map_neighborhood[(map_x, map_y)] = all(p == FREE for p in neighbor_states)
    return latest_map, map_neighborhood, obstacles, free_space, unknown

class Node:
    def __init__(self, v):
        self.v=v
        self.parent=None
        self.child=None


def get_voronoi_regions(obstacles):
    vor = Voronoi(obstacles)
    same_x=[]
    same_y=[]
    same_xy=[]
    vertices=vor.vertices
    n= len(vertices)
    ridges=list(vor.ridge_dict.keys())
    d_min=0.0001#resolution/4.0
    for ridge in ridges:
        if ridge[0] <n and ridge[1]<n:
            p1=vertices[ridge[0]]
            p2 = vertices[ridge[1]]
            dx=abs(p2[0]-p1[0])
            dy=abs(p2[1]-p1[1])
            dxy1=abs(p1[1]-p1[0])
            dxy2 = abs(p2[1] - p2[0])
            if dx < d_min:
                same_x.append(ridge)
            elif dy< d_min:
                same_y.append(ridge)
            elif dxy1<d_min and dxy2 < d_min:
                same_xy.append(ridge)

    nodes=[Node(r[0]) for r in same_x]
    print(ridges)
    # for n in nodes:
    #     u= n.v
    #     v=n.parent
    #     incoming=[p for p in v_vertices if u==p]
    #     outgoing = [p for p in u_vertices if v == p]
    #     if outgoing:
    #         n.child=U[]

def map_call_back(map_message):
    filename = "corridor_map.pickle"
    with open(filename, 'wb') as fp:
        pickle.dump(map_message, fp, protocol=pickle.HIGHEST_PROTOCOL)
    # latest_map, map_neighborhood, obstacles, free_space, unknown = get_map_points(map_message)
    # get_voronoi_regions(obstacles)


if __name__ == '__main__':
    rospy.init_node("voronoi")

    filename = "corridor_map.pickle"
    with open(filename, 'rb') as fp:
        map_message = pickle.load(fp)
        latest_map, map_neighborhood, obstacles, free_space, unknown = get_map_points(map_message)
        get_voronoi_regions(list(latest_map.keys()))



    # rospy.Subscriber('/robot_0/map',OccupancyGrid,map_call_back)
    # rospy.spin()
