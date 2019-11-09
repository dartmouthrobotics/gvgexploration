#!/usr/bin/python

import math
from PIL import Image
import numpy as np
import rospy

WALL_PIXEL = 0
SPACE_PIXEL = 255
class MapAnalyzer:
    def __init__(self, simulator_center,scale,image_pose,image_src):

        size,map_pixels=self.read_map_image(image_src)
        if not map_pixels:
            rospy.loginfo("File not found on path: {}".format(image_src))
            exit(1)
        # image details
        self.map_size = size
        self.map_pixels=map_pixels
        self.world_scale = scale
        self.image_pose = image_pose
        self.world_center = simulator_center

        # difference between center of simulator and that of the map image
        self.dx = self.world_center[0] - image_pose[0]
        self.dy = self.world_center[1] - image_pose[1]

    def read_map_image(self, image_path):
        im = Image.open(image_path, 'r')
        pix_val = list(im.getdata())
        size = im.size
        pixel_values = {}
        for index in range(len(pix_val)):
            i = int(np.floor(index / size[0]))
            j = index % size[0]
            pixel_values[(i, j)] = pix_val[index][0]

        return size, pixel_values

    '''This method takes in a list of world poses and returns the status (obstacle or not) of a given pose using KNN, 
    where K = scale of the map '''
    def get_status_of_poses(self, poses):
        pose_pixel={}
        for pose in poses:
            pixel_pose=self.pose_to_pixel(pose)
            neighbors, pixels= self.get_pixel_neighbors(pixel_pose, distance=self.world_scale)
            black_pixels= [p for p in pixels if p==WALL_PIXEL]
            if len(black_pixels) >= len(pixels)/2:
                pose_pixel[pose]=WALL_PIXEL
            else:
                pose_pixel[pose] = SPACE_PIXEL
        return pose_pixel

    '''
       Convert the world  pose into a pixel position in the image
       :param pose - the x,y coordinate in the map, which is mapped to a pixel in the actual image of the map
       '''

    def pose_to_pixel(self, pose):
        # rescale negative values to start from 0
        rescaled_world_pose_x = pose[0] + self.world_center[0]
        rescaled_world_pose_y = pose[1] + self.world_center[1]

        # convert world pose to pixel positions
        row = self.map_size[1] - (rescaled_world_pose_y + self.dy) * self.world_scale
        col = (rescaled_world_pose_x + self.dx) * self.world_scale
        pixel_pose = (row, col)
        return pixel_pose

    '''
     Convert a pixel position into pixel position into a world pose
     :param pixel_pose - pixel position that shall be converted into a world position
     :return pose - the pose in the world that corresponds to the given pixel position
    '''

    def pixel_to_pose(self, pixel_pose):
        rescaled_world_pose_x = (pixel_pose[1] / self.world_scale) - self.dx
        rescaled_world_pose_y = ((self.map_size[1] - pixel_pose[0]) / self.world_scale) - self.dy

        x = rescaled_world_pose_x - self.world_center[0]
        y = rescaled_world_pose_y - self.world_center[1]
        return (x, y)

    '''
     Get pixel positions of all the neighboring pixels
     :param pixel_pos - center pixel from which to get the neighbors
     :param distance - this defines the width of the neighborhood
     :return neighbors - all valid pixel positions that are neighboring the given pixel within the image

    '''
    def get_pixel_neighbors(self, pixel_pos, distance=1):
        x = pixel_pos[0]
        y = pixel_pos[1]
        neighbors = []
        pixels = []
        for i in range(1, distance + 1):
            east = (x, y + i)
            north = (x, y + i)
            west = (x - i, y)
            south = (x - 1, y)
            ne = (x + i, y + i)
            nw = (x - i, y + i)
            se = (x + i, y - i)
            sw = (x - i, y - i)
            possible_neigbors = [east, north, west, south, ne, nw, se, sw]
            for n in possible_neigbors:
                if (n[1], n[0]) in self.map_pixels:
                    neighbors.append(n)
                    pixels.append(self.map_pixels[n])
        return neighbors, pixels
