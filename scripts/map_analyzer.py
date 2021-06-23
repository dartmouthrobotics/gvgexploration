#!/usr/bin/python
import matplotlib
import os
import signal

matplotlib.use('Agg')
from PIL import Image
import numpy as np
import rospy
from time import sleep
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, pixel2pose, FREE, OCCUPIED, save_data, get_point, scale_down
from gvgexploration.msg import Coverage
from gvgexploration.srv import ExploredRegion, ExploredRegionRequest
from std_msgs.msg import String
from graph import Grid
from nav_msgs.msg import OccupancyGrid
from nav_msgs.srv import GetMap
import rosnode
import tf


class MapAnalyzer:
    def __init__(self):
        self.robot_count = rospy.get_param("/robot_count")
        self.scale = rospy.get_param("/map_scale")
        self.map_file_name = rospy.get_param("/map_file")
        self.run = rospy.get_param("/run")
        self.debug_mode = rospy.get_param("/debug_mode")
        self.max_coverage = rospy.get_param("/max_coverage")
        self.termination_metric = rospy.get_param("/termination_metric")
        self.max_target_info_ratio = rospy.get_param("/max_target_info_ratio")

        self.environment = rospy.get_param("/environment")
        self.method = rospy.get_param("/method")
        self.max_exploration_time = rospy.get_param('/max_exploration_time')
        self.is_active = False
        self.is_raw_map_read = False
        self.total_free_area = 0
        self.current_explored_ratio = 0
        self.map_area = 0
        self.all_coverage_data = []
        self.all_explored_points = set()
        self.all_maps = {}
        self.raw_maps = {}
        self.explored_region = {}
        self.pixel_desc = {}

        self.exploration_start_time = rospy.Time.now().to_sec()

        for i in range(self.robot_count):
            exec("def a_{0}(self, data): self.raw_maps[{0}] = data".format(i))
            exec("setattr(MapAnalyzer, 'callback_map{0}', a_{0})".format(i))
            exec("rospy.Subscriber('/robot_{0}/map', OccupancyGrid, self.callback_map{0},queue_size = 100)".format(i))
            self.all_maps[i] = set()
        self.coverage_pub = rospy.Publisher("/coverage", Coverage, queue_size=10)
        self.shutdown_pub = rospy.Publisher('/shutdown', String, queue_size=2)
        rospy.Subscriber('/latched_map', OccupancyGrid, self.raw_map_handler)
        rospy.Subscriber('/shutdown', String, self.save_all_data)

        # rospy.on_shutdown(self.save_all_data)

    def spin(self):
        r = rospy.Rate(0.2)
        while not rospy.is_shutdown():
            if not self.is_raw_map_read:
                rospy.logerr("Not yet read the raw map")
            else:
                # try:
                self.publish_coverage()
                if self.current_explored_ratio >= self.max_coverage or (
                        rospy.Time.now().to_sec() - self.exploration_start_time) >= self.max_exploration_time * 60:
                    self.shutdown_exploration()

                # except Exception as e:
                #     rospy.logerr("Got this result:::::::: {}".format(e))
            r.sleep()

    def publish_coverage(self):
        common_points = []
        self.all_explored_points.clear()
        for rid in range(self.robot_count):
            self.get_explored_region(rid)
            common_points.append(self.all_maps[rid])
        common_area = set.intersection(*common_points)
        rospy.logerr('New poses explored: {},'.format(len(self.all_explored_points)))
        explored_ratio = len(self.all_explored_points) / self.total_free_area
        common_coverage = len(common_area) / self.total_free_area
        self.current_explored_ratio = explored_ratio
        rospy.logerr("Total points: {}, explored area: {}, common area: {}".format(self.total_free_area,
                                                                                   self.current_explored_ratio,
                                                                                   common_coverage))
        cov_msg = Coverage()
        cov_msg.header.stamp = rospy.Time.now()
        cov_msg.coverage = self.current_explored_ratio
        cov_msg.expected_coverage = -1
        cov_msg.common_coverage = common_coverage
        self.coverage_pub.publish(cov_msg)
        self.all_coverage_data.append(
            {'time': rospy.Time.now().to_sec(), 'explored_ratio': self.current_explored_ratio,
             'common_coverage': common_coverage})

    def get_explored_region(self, rid):
        try:
            if rid in self.raw_maps:
                grid = Grid(self.raw_maps[rid])
                poses = grid.get_explored_region()
                self.all_explored_points.update(poses)
                self.all_maps[rid].clear()
                self.all_maps[rid].update(poses)
        except tf.LookupException as e:
            rospy.logerr(e)

    def get_common_area(self, common_points):
        whole_cells = []
        for cell_set in common_points:
            cells = {get_point(c) for c in cell_set}
            whole_cells.append(cells)
        return whole_cells

    def raw_map_handler(self, occ_grid):
        if not self.is_raw_map_read:
            grid = Grid(occ_grid)
            poses = grid.get_explored_region()
            self.total_free_area = float(len(poses))
            area = occ_grid.info.width * occ_grid.info.height
            rospy.logerr("Free ratio: {}, Width: {}, Height: {}, percentage: {}".format(self.total_free_area,
                                                                                        occ_grid.info.width,
                                                                                        occ_grid.info.height,
                                                                                        self.total_free_area / float(area)))
            self.is_raw_map_read = True

    def check_kill_process(self, pstring):
        for line in os.popen("ps ax | grep " + pstring + " | grep -v grep"):
            fields = line.split()
            pid = fields[0]
            print(line)
            os.kill(int(pid), signal.SIGKILL)

    def save_all_data(self, data):
        save_data(self.all_coverage_data,
                  '{}/coverage_{}_{}_{}_{}_{}.pickle'.format(self.method, self.environment, self.robot_count, self.run,
                                                             self.termination_metric, self.max_target_info_ratio))
        count = self.robot_count
        if self.method == 'recurrent_connectivity':
            count += 1

        sleep(10)
        self.check_kill_process(self.environment)
        all_nodes = []
        count = self.robot_count
        if self.method == 'recurrent_connectivity':
            count += 1
        for i in range(count):
            all_nodes += ['/robot_{}/GetMap'.format(i), '/robot_{}/Mapper'.format(i), '/robot_{}/map_align'.format(i),
                          '/robot_{}/navigator'.format(i), '/robot_{}/operator'.format(i),
                          '/robot_{}/SetGoal'.format(i)]
            if self.method != "gvgexploration":
                all_nodes += ['/robot_{}/explore_client'.format(i), '/robot_{}/graph'.format(i)]

        all_nodes += ['/rosout', '/RVIZ', '/Stage', '/rostopic*']
        rosnode.kill_nodes(all_nodes)
        rospy.signal_shutdown("Exploration complete! Shutting down")

    def shutdown_exploration(self):
        tstr = String()
        tstr.data = "shutdown"
        self.shutdown_pub.publish(tstr)


if __name__ == '__main__':
    rospy.init_node('map_analyzer', anonymous=True)
    rospy.sleep(2)
    analyzer = MapAnalyzer()
    analyzer.spin()
