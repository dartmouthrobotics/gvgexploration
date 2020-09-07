#!/usr/bin/python
import matplotlib

matplotlib.use('Agg')
from PIL import Image
import numpy as np
import rospy
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, pixel2pose, FREE, OCCUPIED, save_data, get_point, scale_down
from gvgexploration.msg import Coverage
from gvgexploration.srv import ExploredRegion, ExploredRegionRequest
from std_msgs.msg import String


class MapAnalyzer:
    def __init__(self):
        rospy.init_node('map_analyzer', anonymous=True)
        self.robot_count = rospy.get_param("~robot_count")
        self.scale = rospy.get_param("~map_scale")
        self.map_file_name = rospy.get_param("~map_file")
        self.run = rospy.get_param("~run")
        self.debug_mode = rospy.get_param("~debug_mode")
        self.termination_metric = rospy.get_param("~termination_metric")
        self.environment = rospy.get_param("~environment")
        self.method = rospy.get_param("~method")
        self.exploration_time = rospy.get_param("~max_exploration_time")
        self.robot_id = rospy.get_param("~robot_id")
        self.is_active = False
        self.total_free_area = 0
        self.free_area_ratio = 0
        self.map_resolution = 0
        self.map_area = 0
        self.all_coverage_data = []
        self.all_explored_points = set()
        self.all_maps = {}
        self.explored_region = {}
        self.pixel_desc = {}
        p = rospy.ServiceProxy('/robot_{}/explored_region'.format(self.robot_id), ExploredRegion)
        p.wait_for_service()
        self.explored_region[self.robot_id] = p
        self.all_maps[self.robot_id] = set()
        # for i in range(self.robot_count):
        #     p = rospy.ServiceProxy('/robot_{}/explored_region'.format(i), ExploredRegion)
        #     p.wait_for_service()
        #     self.explored_region[i] = p
        #     self.all_maps[i] = set()

        self.shutdown_pub = rospy.Publisher("/shutdown".format(self.robot_id), String, queue_size=10)
        self.coverage_pub = rospy.Publisher("/coverage", Coverage, queue_size=10)
        # rospy.Subscriber('/shutdown', String, self.shutdown_callback)
        rospy.on_shutdown(self.save_all_data)

    def spin(self):
        r = rospy.Rate(0.2)
        # self.read_raw_image()
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            try:
                current_time = rospy.Time.now().to_sec()
                elapsed_time = current_time - self.start_time
                if elapsed_time > self.exploration_time * 60:
                    tstr = String()
                    tstr.data = "shutdown"
                    self.shutdown_pub.publish(tstr)
                    rospy.signal_shutdown("Exploration time up! Shutdown")
                if not self.is_active:
                    self.is_active = True
                    self.publish_coverage()
                    self.is_active = False
            except Exception as e:
                rospy.logerr("Got this result:::::::: {}".format(e))
            r.sleep()

    def publish_coverage(self):
        common_points = []
        # for rid in range(self.robot_count):
        self.get_explored_region(self.robot_id)
        common_points.append(self.all_maps[rid])
        common_area = set.intersection(*common_points)
        common_area_size = len(common_area)  # / self.map_area
        explored_area = len(self.all_explored_points)  # / self.map_area
        cov_ratio = explored_area  # / self.free_area_ratio
        common_coverage = common_area_size  # / self.free_area_ratio
        cov_msg = Coverage()
        cov_msg.header.stamp = rospy.Time.now()
        cov_msg.coverage = cov_ratio
        cov_msg.expected_coverage = self.free_area_ratio
        cov_msg.common_coverage = common_coverage
        self.coverage_pub.publish(cov_msg)
        self.all_coverage_data.append(
            {'time': rospy.Time.now().to_sec(), 'explored_ratio': cov_ratio, 'common_coverage': common_coverage,
             'expected_coverage': self.free_area_ratio})

    # def publish_coverage(self):
    #     common_points = []
    #     for rid in range(self.robot_count):
    #         self.get_explored_region(rid)
    #         common_points.append(self.all_maps[rid])
    #     common_area = set.intersection(*common_points)
    #     common_area_size = len(common_area) #/ self.map_area
    #     explored_area = len(self.all_explored_points) #/ self.map_area
    #     cov_ratio = explored_area #/ self.free_area_ratio
    #     common_coverage = common_area_size #/ self.free_area_ratio
    #     cov_msg = Coverage()
    #     cov_msg.header.stamp = rospy.Time.now()
    #     cov_msg.coverage = cov_ratio
    #     cov_msg.expected_coverage = self.free_area_ratio
    #     cov_msg.common_coverage = common_coverage
    #     self.coverage_pub.publish(cov_msg)
    #     self.all_coverage_data.append(
    #         {'time': rospy.Time.now().to_sec(), 'explored_ratio': cov_ratio, 'common_coverage': common_coverage,
    #          'expected_coverage': self.free_area_ratio})

    def get_explored_region(self, rid):
        try:
            explored_points = self.explored_region[rid](ExploredRegionRequest(robot_id=rid))
            poses = explored_points.poses
            self.map_resolution = explored_points.resolution
            self.all_maps[rid].clear()
            self.all_explored_points.clear()
            for p in poses:
                point = (p.position.x, p.position.y)
                self.all_maps[rid].add(point)
                self.all_explored_points.add(point)
        except Exception as e:
            rospy.logerr(e)
            pass

    def get_common_area(self, common_points):
        whole_cells = []
        for cell_set in common_points:
            cells = {get_point(c) for c in cell_set}
            whole_cells.append(cells)
        return whole_cells

    # def read_raw_image(self):
    #     if self.map_file_name:
    #         im = Image.open(self.map_file_name, 'r')
    #         pixelMap = im.load()
    #         free_points = 0
    #         allpixels = 0
    #         width = im.size[0]
    #         height = im.size[1]
    #         self.map_area = float(width * height) / (self.scale ** 2)
    #         for i in range(width):
    #             for j in range(height):
    #                 index = [0.0] * 2
    #                 index[INDEX_FOR_X] = i
    #                 index[INDEX_FOR_Y] = (height - j)
    #                 pixel = pixelMap[i, j]
    #                 allpixels += 1
    #                 if isinstance(pixel, int):
    #                     if pixel > 0:
    #                         free_points += 1
    #                 else:
    #                     pixel = pixelMap[i, j][0]
    #                     if pixel > 0:
    #                         free_points += 1
    #         free_area = float(free_points)
    #         self.free_area_ratio = free_points / float(allpixels)
    #         rospy.logerr("Free ratio: {}, Width: {}, Height: {}, Area: {}: scale: {}".format(self.free_area_ratio, width, height,self.map_area, self.scale))
    #         self.total_free_area = free_area

    # def shutdown_callback(self, msg):
    #     rospy.signal_shutdown('MapAnalyzer: Shutdown command received!')

    def save_all_data(self):
        save_data(self.all_coverage_data,
                  '{}/coverage_{}_{}_{}_{}.pickle'.format(self.method, self.environment, self.robot_count, self.run,
                                                          self.termination_metric))


if __name__ == '__main__':
    analyzer = MapAnalyzer()
    analyzer.spin()
