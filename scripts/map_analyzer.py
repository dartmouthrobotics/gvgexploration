#!/usr/bin/python
import matplotlib

matplotlib.use('Agg')
from PIL import Image
import numpy as np
import rospy
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, pixel2pose, FREE, OCCUPIED, save_data, get_point
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
        self.is_active = False
        self.total_free_area = 0
        self.free_area_ratio = 0
        self.map_resolution = 0
        self.all_coverage_data = []
        self.all_explored_points = set()
        self.all_maps = {}
        self.explored_region = {}
        self.pixel_desc = {}
        for i in range(self.robot_count):
            p = rospy.ServiceProxy('/robot_{}/explored_region'.format(i), ExploredRegion)
            p.wait_for_service()
            rospy.logerr("Service added")
            self.explored_region[i] = p
            self.all_maps[i] = set()

        self.coverage_pub = rospy.Publisher("/coverage", Coverage, queue_size=10)
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)

        rospy.on_shutdown(self.save_all_data)

    def spin(self):
        r = rospy.Rate(0.05)
        self.read_raw_image()
        while not rospy.is_shutdown():
            self.publish_coverage()
            r.sleep()

    def publish_coverage(self):
        if not self.is_active:
            self.is_active = True
            common_points = []
            for rid in range(self.robot_count):
                self.get_explored_region(rid)
                common_points.append(self.all_maps[rid])
            common_area = set.intersection(*common_points)
            scale_val = (self.map_resolution ** 2) / ((1.0 / self.scale) ** 2)
            common_area_size = len(common_area) * scale_val
            explored_area = len(self.all_explored_points) * scale_val  # scale of the map in rviz
            cov_ratio = explored_area / self.total_free_area
            common_coverage = common_area_size / self.total_free_area
            rospy.logerr(
                "Total points: {}, explored area: {}, common area: {}".format(self.total_free_area, cov_ratio,
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
            self.is_active = False

    def get_explored_region(self, rid):
        try:
            explored_points = self.explored_region[rid](ExploredRegionRequest(robot_id=rid))
            poses = explored_points.poses
            self.map_resolution = explored_points.resolution
            for p in poses:
                point = (p.position.x, p.position.y)
                self.all_maps[rid].add(point)
                self.all_explored_points.add(point)
        except Exception as e:
            rospy.logerr(e)
            pass

    def round_point(self, p):
        xc = round(p[INDEX_FOR_X], 2)
        yc = round(p[INDEX_FOR_Y], 2)
        new_p = [0.0] * 2
        new_p[INDEX_FOR_X] = xc
        new_p[INDEX_FOR_Y] = yc
        new_p = tuple(new_p)
        return new_p

    def read_raw_image(self):
        im = Image.open(self.map_file_name, 'r')
        rospy.logerr(self.map_file_name)
        pixelMap = im.load()
        free_points = 0
        allpixels = 0
        width = im.size[0]
        height = im.size[1]
        for i in range(width):
            for j in range(height):
                index = [0.0] * 2
                index[INDEX_FOR_X] = i * 0.1
                index[INDEX_FOR_Y] = (height - j) * 0.1
                # pose = self.round_point(index)
                pixel = pixelMap[i, j]
                allpixels += 1
                if isinstance(pixel, int):
                    if pixel > 0:
                        free_points += 1
                        # self.pixel_desc[pose] = None
                else:
                    pixel = pixelMap[i, j][0]
                    if pixel > 0:
                        free_points += 1
                        # self.pixel_desc[pose] = None
        free_area = float(free_points)
        self.free_area_ratio = free_points / float(allpixels)
        self.total_free_area = free_area
        rospy.logerr(
            "Free pixel ratio: {}, All points: {}, {}".format(self.free_area_ratio, self.total_free_area, im.size))

    def shutdown_callback(self, msg):
        rospy.signal_shutdown('MapAnalyzer: Shutdown command received!')

    def save_all_data(self):
        save_data(self.all_coverage_data,
                  '{}/coverage_{}_{}_{}_{}.pickle'.format(self.method, self.environment, self.robot_count, self.run,
                                                          self.termination_metric))


if __name__ == '__main__':
    analyzer = MapAnalyzer()
    analyzer.spin()
