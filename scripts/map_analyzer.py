#!/usr/bin/python
import matplotlib
import os
import signal
matplotlib.use('Agg')
from PIL import Image
import numpy as np
import rospy
from time import sleep
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, pixel2pose, FREE, OCCUPIED, save_data, get_point,scale_down
from gvgexploration.msg import Coverage
from gvgexploration.srv import ExploredRegion, ExploredRegionRequest
from std_msgs.msg import String
from graph import Grid
from nav_msgs.msg import OccupancyGrid
import rosnode
import tf


class MapAnalyzer:
    def __init__(self):
        rospy.init_node('map_analyzer', anonymous=True)
        self.robot_count = rospy.get_param("/robot_count")
        self.scale = rospy.get_param("/map_scale")
        self.map_file_name = rospy.get_param("/map_file")
        self.run = rospy.get_param("/run")
        self.debug_mode = rospy.get_param("/debug_mode")
        self.max_coverage=rospy.get_param("/max_coverage")
        self.termination_metric = rospy.get_param("/termination_metric")
        self.environment = rospy.get_param("/environment")
        self.method = rospy.get_param("/method")
        self.is_active = False
        self.total_free_area = 0
        self.free_area_ratio = 0
        self.map_area = 0
        self.all_coverage_data = []
        self.all_explored_points = set()
        self.all_maps = {}
        self.explored_region = {}
        self.pixel_desc = {}
        self.raw_maps= {}

        for i in range(self.robot_count):
            exec("def a_{0}(self, data): self.raw_maps[{0}] = data".format(i))
            exec("setattr(MapAnalyzer, 'callback_map{0}', a_{0})".format(i))
            exec("rospy.Subscriber('/robot_{0}/map', OccupancyGrid, self.callback_map{0},queue_size = 100)".format(i))
            self.all_maps[i] = set()

        self.coverage_pub = rospy.Publisher("/coverage", Coverage, queue_size=10)
        self.shutdown_pub=rospy.Publisher('/shutdown', String,queue_size=2 )
        rospy.Subscriber('/shutdown', String,self.save_all_data)

        # rospy.on_shutdown(self.save_all_data)

    def spin(self):
        r = rospy.Rate(0.2)
        self.read_raw_image()
        while not rospy.is_shutdown():
            try:
                if not self.is_active:
                    self.is_active = True
                    self.publish_coverage()
                    self.is_active = False
            except Exception as e:
                rospy.logerr("Got this result:::::::: {}".format(e))
            r.sleep()

    def publish_coverage(self):
        common_points = []
        for rid in range(self.robot_count):
            self.get_explored_region(rid)
            common_points.append(self.all_maps[rid])
        common_area = set.intersection(*common_points)
        # TODO cleanup
        common_area_size = len(common_area) #/ self.map_area
        explored_area = len(self.all_explored_points) #/ self.map_area
        cov_ratio = explored_area / self.total_free_area
        common_coverage = common_area_size / self.total_free_area
        rospy.logerr("Total points: {}, explored area: {}, common area: {}".format(self.total_free_area, cov_ratio,
                                                                                   common_coverage))
        cov_msg = Coverage()
        cov_msg.header.stamp = rospy.Time.now()
        cov_msg.coverage = cov_ratio
        cov_msg.expected_coverage = self.free_area_ratio
        cov_msg.common_coverage = common_coverage
        self.coverage_pub.publish(cov_msg)
        self.all_coverage_data.append(
            {'time': rospy.Time.now().to_sec(), 'explored_ratio': cov_ratio, 'common_coverage': common_coverage,
             'expected_coverage': self.free_area_ratio})
        if cov_ratio >= self.max_coverage:
            self.shutdown_exploration()

    def get_explored_region(self, rid):
        try:
            if rid in self.raw_maps:
                grid=Grid(self.raw_maps[rid])
                poses = grid.get_explored_region()
                rospy.logerr("len of poses for {} is {}".format(rid, len(poses)))
                self.all_maps[rid].clear()
                self.all_explored_points.clear()
                for p in poses:
                    point=tuple(p)
                    self.all_maps[rid].add(point)
                    self.all_explored_points.add(point)
        except tf.LookupException as e:
            rospy.logerr(e)

    def get_common_area(self, common_points):
        whole_cells = []
        for cell_set in common_points:
            cells = {get_point(c) for c in cell_set}
            whole_cells.append(cells)
        return whole_cells

    def read_raw_image(self):
        # TODO reading from the yaml file.
        # TODO adjust according to the same resolution as the grid, so that calculations are correct.
        if self.map_file_name:
            im = Image.open(self.map_file_name, 'r')
            pixelMap = im.load()
            free_points = 0
            allpixels = 0
            width = im.size[0]
            height = im.size[1]
            self.map_area = float(width * height) / (self.scale ** 2)
            for i in range(width):
                for j in range(height):
                    index = [0.0] * 2
                    index[INDEX_FOR_X] = i
                    index[INDEX_FOR_Y] = (height - j)
                    pixel = pixelMap[i, j]
                    allpixels += 1
                    if isinstance(pixel, int):
                        if pixel > 0:
                            free_points += 1
                    else:
                        pixel = pixelMap[i, j][0]
                        if pixel > 0:
                            free_points += 1
            free_area = float(free_points)
            self.free_area_ratio = free_points / float(allpixels)
            rospy.logerr("Free ratio: {}, Width: {}, Height: {}, Area: {}: scale: {}".format(self.free_area_ratio, width, height,self.map_area, self.scale))
            self.total_free_area = free_area


    def check_kill_process(self,pstring):
        for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
            fields = line.split()
            pid = fields[0]
            print(line)
            os.kill(int(pid), signal.SIGKILL)

    def save_all_data(self,data):
        save_data(self.all_coverage_data,'{}/coverage_{}_{}_{}_{}.pickle'.format(self.method, self.environment, self.robot_count, self.run,self.termination_metric))

    def shutdown_exploration(self):
        tstr = String()
        tstr.data = "shutdown"
        self.shutdown_pub.publish(tstr)
        sleep(5)
        self.check_kill_process(self.environment)
        all_nodes=[]
        for i in range(self.robot_count):
            all_nodes+=['/robot_{}/GetMap'.format(i),'/robot_{}/Mapper'.format(i),'/robot_{}/map_align'.format(i),'/robot_{}/navigator'.format(i),'/robot_{}/operator'.format(i),'/robot_{}/SetGoal'.format(i)]

        all_nodes+=['/rosout','/RVIZ','/Stage','/rostopic*']
        rosnode.kill_nodes(all_nodes)
        rospy.signal_shutdown("Exploration complete! Shutting down")



if __name__ == '__main__':
    analyzer = MapAnalyzer()
    analyzer.spin()
