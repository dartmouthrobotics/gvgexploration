#!/usr/bin/python

from PIL import Image
import numpy as np
import rospy
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, pixel2pose, FREE, OCCUPIED, save_data, get_point
from nav_msgs.msg import OccupancyGrid
from gvgexploration.msg import Coverage
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
        self.total_free_area = 0
        self.free_area_ratio = 0
        self.map_resolution = 0
        self.all_coverage_data = []
        self.all_explored_points = set()
        self.all_maps = {}
        self.robot_maps = {}

        for i in range(self.robot_count):
            exec("def a_{0}(self, data): self.robot_maps[{0}] = data".format(i))
            exec("setattr(MapAnalyzer, 'callback_map_teammate{0}', a_{0})".format(i))
            exec("rospy.Subscriber('/robot_{0}/map', OccupancyGrid, self.callback_map_teammate{0}, "
                 "queue_size = 10)".format(i))

        self.coverage_pub = rospy.Publisher("/coverage", Coverage, queue_size=10)
        rospy.Subscriber('/shutdown', String, self.shutdown_callback)
        rospy.on_shutdown(self.save_all_data)

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            self.publish_coverage()
            r.sleep()

    def publish_coverage(self):
        common_points = []
        if len(self.robot_maps) == self.robot_count:
            map0 = self.robot_maps[0]
            origin_x0 = map0.info.origin.position.x
            origin_y0 = map0.info.origin.position.y
            for rid in range(self.robot_count):
                map = self.robot_maps[rid]  # rospy.wait_for_message('/robot_{}/map'.format(rid), OccupancyGrid)
                origin_pos = map.info.origin.position
                origin_x = origin_pos.x - (origin_pos.x - origin_x0)
                origin_y = origin_pos.y - (origin_pos.y - origin_y0)
                cov_set, unexplored_set = self.get_map_description(map, rid, origin_x, origin_y)
                if rid not in self.all_maps:
                    self.all_maps[rid] = cov_set
                else:
                    self.all_maps[rid] = self.all_maps[rid].union(cov_set)
                common_points.append(self.all_maps[rid])
                self.all_explored_points = self.all_explored_points.union(cov_set)
            common_area = set.intersection(*common_points)
            common_area_size = len(common_area) * 0.25
            explored_area = len(self.all_explored_points) * 0.25  # scale of the map in rviz
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

    def get_map_description(self, occ_grid, rid, origin_x, origin_y):
        resolution = occ_grid.info.resolution
        if not self.map_resolution:
            self.map_resolution = round(resolution, 2)
            self.read_raw_image()
        # origin_pos = occ_grid.info.origin.position
        # rospy.logerr(
        #     "ID: {}, current origin: {}, original: {}".format(rid, (origin_x, origin_y), (origin_pos.x, origin_pos.y)))
        height = occ_grid.info.height
        width = occ_grid.info.width
        grid_values = np.array(occ_grid.data).reshape((height, width)).astype(np.float32)
        num_rows = grid_values.shape[0]
        num_cols = grid_values.shape[1]
        explored_poses = set()
        unexplored_poses = set()
        for row in range(num_rows):
            for col in range(num_cols):
                index = [0] * 2
                index[INDEX_FOR_Y] = num_rows - row - 1
                index[INDEX_FOR_X] = col
                index = tuple(index)
                pose = pixel2pose(index, origin_x, origin_y, self.map_resolution)
                pose = self.round_point(pose)
                p = grid_values[num_rows - row - 1, col]
                if p == FREE:
                    explored_poses.add(pose)
                else:
                    unexplored_poses.add(pose)
        return explored_poses, unexplored_poses

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
        for i in range(im.size[0]):
            for j in range(im.size[1]):
                pixel = pixelMap[i, j]
                allpixels += 1
                if isinstance(pixel, int):
                    if pixel > 0:
                        free_points += 1
                else:
                    pixel = pixelMap[i, j][0]
                    if pixel > 0:
                        free_points += 1
        self.total_free_area = free_points  # float(allpixels)
        self.free_area_ratio = free_points / self.total_free_area
        rospy.logerr(
            "Free pixel ratio: {}, All points: {}, {}".format(self.free_area_ratio, self.total_free_area, im.size))

    def shutdown_callback(self, msg):
        rospy.signal_shutdown('MapAnalyzer: Shutdown command received!')

    def save_all_data(self):
        save_data(self.all_coverage_data,
                  'gvg/coverage_{}_{}_{}_{}.pickle'.format(self.environment, self.robot_count, self.run,
                                                           self.termination_metric))


if __name__ == '__main__':
    analyzer = MapAnalyzer()
    analyzer.spin()
