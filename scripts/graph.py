#!/usr/bin/python

import time
import numpy as np
from numpy.linalg import norm
from scipy.spatial import Voronoi, voronoi_plot_2d
import message_filters
import igraph
from bresenham import bresenham
from std_msgs.msg import ColorRGBA
import copy
import matplotlib
import heapq

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rospy
from threading import Lock
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, PointStamped
from gvgexploration.msg import *
from gvgexploration.srv import *
import project_utils as pu
from project_utils import euclidean_distance
from graham_scan import graham_scan
from nav2d_navigator.msg import *
from project_utils import INDEX_FOR_X, INDEX_FOR_Y, save_data
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from tf import TransformListener
from geometry_msgs.msg import Point
from scipy.ndimage import minimum_filter

from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan

INF = 100000
SCALE = 10
FREE = 0.0
OCCUPIED = 90.0
UNKNOWN = -1.0
SMALL = 0.000000000001
FONT_SIZE = 16
MARKER_SIZE = 12


class PQueue:
    def __init__(self):
        self._data = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._data, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._data)[-1]

    def size(self):
        return len(self._data)

    def items(self):
        return [d[-1] for d in self._data]


class Grid:
    """Occupancy Grid."""

    def __init__(self, map_msg):
        self.header = map_msg.header
        self.origin_translation = [map_msg.info.origin.position.x,
                                   map_msg.info.origin.position.y, map_msg.info.origin.position.z]
        self.origin_quaternion = [map_msg.info.origin.orientation.x,
                                  map_msg.info.origin.orientation.y,
                                  map_msg.info.origin.orientation.z,
                                  map_msg.info.origin.orientation.w]
        self.grid = np.reshape(map_msg.data,
                               (map_msg.info.height,
                                map_msg.info.width))  # shape: 0: height, 1: width.
        self.resolution = map_msg.info.resolution  # cell size in meters.

        self.tf_listener = TransformListener()  # Transformation listener.

        self.transformation_matrix_map_grid = self.tf_listener.fromTranslationRotation(
            self.origin_translation,
            self.origin_quaternion)

        self.transformation_matrix_grid_map = np.linalg.inv(self.transformation_matrix_map_grid)

        # Array to check neighbors.
        self.cell_radius = int(10 / self.resolution)  # TODO parameter
        self.footprint = np.ones((self.cell_radius + 1, self.cell_radius + 1))
        # self.plot()

    def plot(self):
        fig1 = plt.gcf()
        plt.imshow(self.grid)
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.xlabel('xlabel', fontsize=18)
        plt.ylabel('ylabel', fontsize=16)
        fig1.savefig('map.png', dpi=100)

    def cell_at(self, x, y):
        """Return cell value at x (column), y (row)."""
        return self.grid[int(y), int(x)]
        # return self.grid[x + y * self.width]

    def is_free(self, x, y):
        if self.within_boundaries(x, y):
            return 0 <= self.cell_at(x, y) < 50  # TODO: set values.
        else:
            return False

    def is_obstacle(self, x, y):
        if self.within_boundaries(x, y):
            return self.cell_at(x, y) >= 50  # TODO: set values.
        else:
            return False

    def is_unknown(self, x, y):
        if self.within_boundaries(x, y):
            return self.cell_at(x, y) < 0  # TODO: set values.
        else:
            return True

    def within_boundaries(self, x, y):
        if 0 <= y < self.grid.shape[0] and 0 <= x < self.grid.shape[1]:
            return True
        else:
            return False

    def convert_coordinates_i_to_xy(self, i):
        """Convert coordinates if the index is given on the flattened array."""
        x = i % self.grid.shape[1]  # col
        y = i / self.grid.shape[1]  # row
        return x, y

    def wall_cells(self):
        """
        Return only *wall cells* -- i.e. obstacle cells that have free or 
            unknown cells as neighbors -- as columns and rows.
        """

        # Array to check neighbors.
        window = np.asarray([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ])
        # Return all cells that are obstacles and satisfy the neighbor_condition
        neighbor_condition = self.grid > minimum_filter(self.grid,
                                                        footprint=window, mode='constant', cval=10000)  # TODO: value.
        obstacles = np.nonzero((self.grid >= OCCUPIED) & neighbor_condition)
        obstacles = np.stack((obstacles[1], obstacles[0]),
                             axis=1)  # 1st column x, 2nd column, y.

        return obstacles

    def is_frontier(self, previous_cell, current_cell,
                    distance, fc_count=1):
        if fc_count == 0:
            rospy.logerr("FRONTIER CHECK: PASSED")
            return True
        """current_cell a frontier?"""
        v = previous_cell - current_cell
        u = v / np.linalg.norm(v)

        end_cell = current_cell - distance * u

        x1, y1 = current_cell.astype(int)
        x2, y2 = end_cell.astype(int)

        for p in list(bresenham(x1, y1, x2, y2)):
            if self.is_unknown(*p):
                return True
            elif self.is_obstacle(*p):
                return False
        return False

    def get_newinfo(self, previous_cell, current_cell, distance):
        """Number of unknown cells in the region  @Kizito"""
        v = previous_cell - current_cell
        u = v / np.linalg.norm(v)
        end_cell = current_cell - distance * u
        x1, y1 = current_cell.astype(int)
        x2, y2 = end_cell.astype(int)
        unknown_cells = 0
        all_cells = 1.0
        for p in list(bresenham(x1, y1, x2, y2)):
            all_cells += 1
            if self.is_unknown(*p):
                unknown_cells += 1
        return unknown_cells / all_cells

    def get_path_obstacles(self, start_cell, end_cell):
        """Number of unknown cells in the region  @Kizito"""
        x1, y1 = start_cell.astype(int)
        x2, y2 = end_cell.astype(int)
        obstacle_cells = 0
        for p in list(bresenham(x1, y1, x2, y2)):
            if self.is_obstacle(*p):
                obstacle_cells += 1
        return obstacle_cells

    def line_in_unknown(self, start_cell, end_cell):
        x1, y1 = start_cell.astype(int)
        x2, y2 = end_cell.astype(int)

        unknown_cells_counter = 0
        for p in list(bresenham(x1, y1, x2, y2)):
            if self.is_unknown(*p):
                unknown_cells_counter += 1
            elif self.is_obstacle(*p):
                return False
        return unknown_cells_counter

    def unknown_area_approximate(self, cell):
        """Approximate unknown area with the robot at cell."""
        cell_x = int(cell[INDEX_FOR_X])
        cell_y = int(cell[INDEX_FOR_Y])

        min_x = np.max((0, cell_x - self.cell_radius))
        max_x = np.min((self.grid.shape[1], cell_x + self.cell_radius + 1))
        min_y = np.max((0, cell_y - self.cell_radius))
        max_y = np.min((self.grid.shape[0], cell_y + self.cell_radius + 1))

        return (self.grid[min_y:max_y, min_x:max_x] < FREE).sum()

    def unknown_area(self, cell):  # TODO orientation of the robot if fov is not 360 degrees
        """Return unknown area with the robot at cell"""
        unknown_cells = set()

        shadow_angle = set()
        cell_x = int(cell[INDEX_FOR_X])
        cell_y = int(cell[INDEX_FOR_Y])
        for d in np.arange(1, self.cell_radius):  # TODO orientation
            for x in xrange(cell_x - d, cell_x + d + 1):  # go over x axis
                for y in range(cell_y - d, cell_y + d + 1):  # go over y axis
                    if self.within_boundaries(x, y):
                        angle = np.around(np.rad2deg(pu.theta(cell, [x, y])), decimals=1)  # TODO parameter
                        if angle not in shadow_angle:
                            if self.is_obstacle(x, y):
                                shadow_angle.add(angle)
                            elif self.is_unknown(x, y):
                                unknown_cells.add((x, y))

        return len(unknown_cells)

    def pose_to_grid(self, pose):
        """Pose (x,y) in header.frame_id to grid coordinates"""
        # Get transformation matrix map-occupancy grid.
        return (self.transformation_matrix_grid_map.dot([pose[0], pose[1], 0, 1]))[
               0:2] / self.resolution  # TODO check int.

    def grid_to_pose(self, grid_coordinate):
        """Pose (x,y) in grid coordinates to pose in frame_id"""
        # Get transformation matrix map-occupancy grid.
        return (self.transformation_matrix_map_grid.dot(
            np.array([grid_coordinate[0] * self.resolution,
                      grid_coordinate[1] * self.resolution, 0, 1])))[0:2]

    def get_explored_region(self):
        """ Get all the explored cells on the grid map"""

        # TODO refactor the code, right now hardcoded to quickly solve the problem of non-common areas.
        # TODO more in general use matrices.
        def nearest_multiple(number, res=0.2):
            return np.round(res * np.floor(round(number / res, 2)), 1)

        p_in_sender = PointStamped()
        p_in_sender.header = self.header

        poses = set()
        self.tf_listener.waitForTransform("robot_0/map",
                                          self.header.frame_id, rospy.Time(),
                                          rospy.Duration(4.0))
        p_in_sender.header.stamp = rospy.Time()
        for x in range(self.grid.shape[1]):
            for y in range(self.grid.shape[0]):
                if self.is_free(x, y):
                    p = self.grid_to_pose((x, y))
                    p_in_sender.point.x = p[0]
                    p_in_sender.point.y = p[1]

                    p_in_common_ref_frame = self.tf_listener.transformPoint("robot_0/map",
                                                                            p_in_sender).point
                    poses.add((nearest_multiple(p_in_common_ref_frame.x), nearest_multiple(p_in_common_ref_frame.y)))
        return poses


class Graph:
    def __init__(self):
        # Robot ID.
        self.robot_id = rospy.get_param("~robot_id")

        # Parameters related to the experiment.
        self.robot_count = rospy.get_param("/robot_count")
        self.environment = rospy.get_param("/environment")
        self.run = rospy.get_param("/run")
        self.bs_pose = rospy.get_param("/bs_pose")
        self.img_res = rospy.get_param("/image_resolution")
        self.target_distance = rospy.get_param("/target_distance")

        # Physical parameters of the robot.
        self.robot_radius = rospy.get_param('/robot_{}/robot_radius'.format(self.robot_id))  # meter.

        # Internal data.
        self.latest_map = None  # Grid map.

        self.tf_listener = TransformListener()  # Transformation listener.

        # ROS publisher
        self.marker_colors = rospy.get_param('/robot_colors')
        self.marker_pub = rospy.Publisher('voronoi', Marker, queue_size=0)  # log.
        self.scan_marker_pub = rospy.Publisher('scan_voronoi', Marker, queue_size=0)  # log.
        self.border_edge_pub = rospy.Publisher('border_edge', Marker, queue_size=0)
        self.gate_marker_pub = rospy.Publisher('gate', Marker, queue_size=0)  # log.

        self.marker_colors = rospy.get_param('/robot_colors')
        self.alpha = [1.0, 0.5, 0]

        # TO CHECK WHAT IS NEEDED.
        self.min_hallway_width = None
        self.pixel_desc = {}
        self.all_poses = set()
        self.obstacles = {}
        self.free_points = {}
        self.map_resolution = 0.05
        self.plot_intersection_active = False
        self.plot_data_active = False
        self.is_first_gate_request = True
        self.lock = Lock()

        self.obstacles = {}
        self.adj_list = {}
        self.edges = {}
        self.leaf_slope = {}
        self.longest = None
        self.adj_dict = {}
        self.tree_size = {}
        self.leave_dict = {}
        self.parent_dict = {}
        self.new_information = {}
        self.known_points = {}
        self.unknown_points = {}
        self.global_leaves = {}
        self.leaf_edges = {}
        self.leaf_obstacles = {}
        self.performance_data = []
        self.explored_points = set()
        self.last_intersection = None

        self.prev_ridge = None

        self.debug_mode = rospy.get_param("/debug_mode")
        self.method = rospy.get_param("/method")
        self.bs_pose = rospy.get_param('/bs_pose')
        self.map_scale = rospy.get_param('/map_scale')
        self.graph_scale = rospy.get_param("/graph_scale")
        self.termination_metric = rospy.get_param("/termination_metric")
        self.max_target_info_ratio = rospy.get_param("/max_target_info_ratio")
        self.frontier_threshold = rospy.get_param("/frontier_threshold")
        self.min_hallway_width = rospy.get_param("/min_hallway_width")  # * self.graph_scale
        self.comm_range = rospy.get_param("/comm_range")  # * self.graph_scale
        self.point_precision = rospy.get_param("/point_precision")

        self.scan_subscriber = message_filters.Subscriber('filtered_scan', LaserScan)
        self.obstacle_pose_pub = rospy.Publisher('laser_obstacles', Marker, queue_size=0)

        self.lidar_scan_radius = rospy.get_param("/lidar_scan_radius")  # * self.graph_scale
        self.lidar_fov = rospy.get_param("/lidar_fov")
        self.slope_bias = rospy.get_param("/slope_bias")
        self.separation_bias = rospy.get_param("/separation_bias")  # * self.graph_scale
        self.opposite_vector_bias = rospy.get_param("/opposite_vector_bias")
        rospy.Service('/robot_{}/rendezvous'.format(self.robot_id), RendezvousPoints,
                      self.fetch_rendezvous_points_handler)
        rospy.Service('/robot_{}/explored_region'.format(self.robot_id), ExploredRegion,
                      self.fetch_explored_region_handler)
        rospy.Service('/robot_{}/frontier_points'.format(self.robot_id), FrontierPoint, self.frontier_point_handler)

        rospy.Service('/robot_{}/graph_distance'.format(self.robot_id), GraphDistance, self.graph_distance_handler)

        # rospy.Subscriber('/robot_{}/map'.format(self.robot_id),OccupancyGrid, self.map_callback)
        rospy.Subscriber('/shutdown', String, self.save_all_data)

        rospy.wait_for_service('/robot_{}/static_map'.format(self.robot_id))
        self.get_map = rospy.ServiceProxy('/robot_{}/static_map'.format(self.robot_id), GetMap)

        # for online gvg generation @Kizito
        self.process_scan = True
        self.augmented_graph = None
        self.other_leaves = []
        self.gvg_exp_q = PQueue()

        # ====== ends here ========== @Kizito

        self.already_shutdown = False
        self.robot_pose = None

        self.assigned_frontier_leaves = PQueue()
        self.assigned_visited_leaves = {}
        self.assigned_visited_leaves = {}
        self.fc_count = 0
        if self.robot_id != 0:
            self.fc_count = 1
        # edges
        self.deleted_nodes = {}
        self.deleted_obstacles = {}
        self.last_graph_update_time = rospy.Time.now().to_sec()
        # rospy.on_shutdown(self.save_all_data)
        rospy.loginfo('Robot {}: Successfully created graph node'.format(self.robot_id))

    def compute_gvg(self):
        """Compute GVG for exploration."""

        start_time_clock = time.clock()
        # Get only wall cells for the Voronoi.
        obstacles = self.latest_map.wall_cells()
        end_time_clock = time.clock()
        pu.log_msg(self.robot_id, "generate obstacles2 {}".format(end_time_clock - start_time_clock), self.debug_mode)

        start_time_clock = time.clock()
        # Get Voronoi diagram.
        vor = Voronoi(obstacles)

        end_time_clock = time.clock()
        pu.log_msg(self.robot_id, "voronoi {}".format(end_time_clock - start_time_clock), self.debug_mode)
        start_time_clock = time.clock()
        # Initializing the graph.
        self.graph = igraph.Graph()
        # Correspondance between graph vertex and Voronoi vertex IDs.
        voronoi_graph_correspondance = {}

        # Simplifying access of Voronoi data structures.
        vertices = vor.vertices
        ridge_vertices = vor.ridge_vertices
        ridge_points = vor.ridge_points

        edges = []
        weights = []
        # Create a graph based on ridges.
        for i in xrange(len(ridge_vertices)):
            ridge_vertex = ridge_vertices[i]
            # If any of the ridge vertices go to infinity, then don't add.
            if ridge_vertex[0] == -1 or ridge_vertex[1] == -1:
                continue
            p1 = vertices[ridge_vertex[0]]
            p2 = vertices[ridge_vertex[1]]

            # Obstacle points determining the ridge.
            ridge_point = ridge_points[i]
            q1 = obstacles[ridge_point[0]]
            q2 = obstacles[ridge_point[1]]

            # If the vertices on the ridge are in the free space
            # and distance between obstacle points is large enough for the robot
            if self.latest_map.is_free(p1[INDEX_FOR_X], p1[INDEX_FOR_Y]) and \
                    self.latest_map.is_free(p2[INDEX_FOR_X], p2[INDEX_FOR_Y]) and \
                    euclidean_distance(q1, q2) > 4 * self.min_edge_length:

                # Add vertex and edge.
                graph_vertex_ids = [-1, -1]  # temporary for finding verted IDs.

                # Determining graph vertex ID if existing or not.
                for point_id in xrange(len(graph_vertex_ids)):
                    if ridge_vertex[point_id] not in voronoi_graph_correspondance:
                        # if not existing, add new vertex.
                        graph_vertex_ids[point_id] = self.graph.vcount()
                        self.graph.add_vertex(coord=vertices[ridge_vertex[point_id]], name=graph_vertex_ids[point_id])
                        voronoi_graph_correspondance[ridge_vertex[point_id]] = graph_vertex_ids[point_id]
                    else:
                        # Otherwise, already added before.
                        graph_vertex_ids[point_id] = voronoi_graph_correspondance[ridge_vertex[point_id]]

                # Add edge.
                self.graph.add_edge(graph_vertex_ids[0], graph_vertex_ids[1],
                                    weight=euclidean_distance(p1, p2))
            else:
                # Otherwise, edge not added.
                continue

        # Take only the largest component.
        cl = self.graph.clusters()
        self.graph = cl.giant()

        # self.get_adjacency_list(self.edges)
        # self.connect_subtrees()
        # self.graph=self.merge_similar_edges()
        self.prune_leaves()

        end_time_clock = time.clock()
        pu.log_msg(self.robot_id, "ridge {}".format(end_time_clock - start_time_clock), self.debug_mode)

        # Publish GVG.
        self.publish_edges()

    def count_path_vertices(self, vid1, vid2):
        result = self.graph.shortest_paths(source=[vid1], target=[vid2])
        # pu.log_msg(self.robot_id,"Edge connectivity between {} and {}: {}".format(vid1,vid2,result),self.debug_mode)
        return result[0][0]

    def merge_similar_edges2(self):
        current_vertex_id = 0  # traversing graph from vertex 0.
        self.leaves = {}
        while current_vertex_id < self.graph.vcount():
            if self.graph.degree(current_vertex_id) == 2:
                p_id, r_id = self.graph.neighbors(current_vertex_id)
                p = self.graph.vs["coord"][p_id]
                current_vertex = self.graph.vs["coord"][current_vertex_id]
                r = self.graph.vs["coord"][r_id]

                slope_pc = pu.get_slope(p, current_vertex)
                slope_cr = pu.get_slope(p, current_vertex)
                if np.isclose(slope_pc,
                              slope_cr, rtol=0.001, atol=0.001, equal_nan=True) or \
                        np.isclose(slope_cr, slope_pc, rtol=0.001, atol=0.001,
                                   equal_nan=True):  # TODO tune.
                    pq_id = self.graph.get_eid(p_id, current_vertex_id)
                    qr_id = self.graph.get_eid(current_vertex_id, r_id)

                    self.graph.add_edge(p_id, r_id,
                                        weight=self.graph.es["weight"][pq_id] + self.graph.es["weight"][qr_id])
                    self.graph.delete_vertices(current_vertex_id)
                else:

                    current_vertex_id += 1
            else:
                if self.graph.degree(current_vertex_id) == 1:
                    self.leaves[current_vertex_id] = self.latest_map.unknown_area_approximate(current_vertex)
                current_vertex_id += 1

    # Un referenced method being modified by Kizito.
    def prune_leaves2(self, graph):
        # self.leaves = {}
        # self.intersections = {}
        # current_vertex_id=0
        for v in graph.vs:
            current_vertex_id = v.index
            if graph.degree(current_vertex_id) == 1:
                neighbor_id = graph.neighbors(current_vertex_id)[0]
                neighbor = graph.vs["coord"][neighbor_id]
                current_vertex = graph.vs["coord"][current_vertex_id]
                if not self.latest_map.is_frontier(
                        neighbor, current_vertex, 2 * self.min_range_radius):  # TODO parameter.
                    self.graph.delete_vertices(current_vertex_id)
                else:
                    self.leaves[current_vertex_id] = self.latest_map.unknown_area_approximate(current_vertex)
            # else:
            #     if graph.degree(current_vertex_id) > 2:
            #         self.intersections[current_vertex_id] = graph.vs["coord"][current_vertex_id]
        # self.publish_leaves()

    def merge_similar_edges(self):
        graph_cp = copy.deepcopy(self.graph)

        def _vid(v_name):
            return list(graph_cp.vs(name=v_name))[0].index

        def _vname(vid):
            return graph_cp.vs["name"][vid]

        def _vcoord(vname):
            return graph_cp.vs["coord"][_vid(vname)]

        first_node = _vname(list(graph_cp.vs)[0].index)
        parents = {first_node: None}
        neis = graph_cp.neighbors(_vid(first_node))
        if len(neis) > 1:
            parents[first_node] = _vname(neis[0])
            parents[parents[first_node]] = None
        deleted_nodes = {}
        S = [first_node]
        visited = {}
        while len(S) > 0:
            u = S.pop()
            if u not in deleted_nodes:
                all_neis = graph_cp.neighbors(_vid(u))
                neighbors = [_vname(k) for k in all_neis if _vname(k) != parents[u]]
                if len(neighbors) == 1:
                    v = neighbors[0]
                    if v not in visited:
                        S.append(v)
                        if parents[u]:
                            us = pu.get_vector(_vcoord(parents[u]), _vcoord(u))
                            ps = pu.get_vector(_vcoord(u), _vcoord(v))
                            cos_theta, separation = pu.compute_similarity(us, ps, (_vcoord(parents[u]), _vcoord(u)),
                                                                          (_vcoord(u), _vcoord(v)))
                            if 1 - self.opposite_vector_bias <= cos_theta <= 1:
                                parents[v] = parents[u]
                                deleted_nodes[u] = None
                                graph_cp.add_edges([(_vid(parents[u]), _vid(v))])
                                graph_cp.delete_edges([(_vid(u), _vid(v))])
                                graph_cp.delete_edges([(_vid(parents[u]), _vid(u))])
                                rospy.logerr("Deleted edge..")
                            else:
                                parents[v] = u
                                # rospy.logerr("Parents: {}".format(parents))
                        else:
                            parents[v] = u
                else:
                    for v in neighbors:
                        if v not in visited:
                            S.append(v)
                            parents[v] = u
                visited[u] = None
        return graph_cp

    def prune_leaves(self):
        current_vertex_id = 0  # traversing graph from vertex 0.
        self.leaves = {}
        self.intersections = {}
        while current_vertex_id < self.graph.vcount():
            if self.graph.degree(current_vertex_id) == 1:
                neighbor_id = self.graph.neighbors(current_vertex_id)[0]
                neighbor = self.graph.vs["coord"][neighbor_id]
                current_vertex = self.graph.vs["coord"][current_vertex_id]
                if not self.latest_map.is_frontier(neighbor, current_vertex, self.min_range_radius,
                                                   fc_count=self.fc_count):  # TODO parameter.
                    self.graph.delete_vertices(current_vertex_id)
                else:
                    self.leaves[current_vertex_id] = self.latest_map.unknown_area_approximate(current_vertex)
                    current_vertex_id += 1
            else:
                #     if self.graph.degree(current_vertex_id) > 2:
                #         self.intersections[current_vertex_id] = self.graph.vs["coord"][current_vertex_id]
                current_vertex_id += 1
        # self.publish_leaves()

    def publish_edges(self):
        """For debug, publishing of GVG."""

        # Marker that will contain line sequences.
        m = Marker()
        m.id = 0
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.LINE_LIST
        # TODO constant values set at the top.
        m.color.a = 1.0
        m.color.r = 1.0
        m.scale.x = 0.1

        # Plot each edge.
        for edge in self.graph.get_edgelist():
            for vertex_id in edge:
                # Grid coordinate for the vertex.
                p = self.graph.vs["coord"][vertex_id]
                p_t = self.latest_map.grid_to_pose(p)
                p_ros = Point(x=p_t[0], y=p_t[1])

                m.points.append(p_ros)

        # Publish marker.
        self.marker_pub.publish(m)

    def publish_current_vertex(self, vertex_id, marker_id=1):
        # Publish current vertex
        rid = str(self.robot_id)
        m = Marker()
        m.id = marker_id
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.SPHERE
        m.color.a = 1.0
        m.color.r = self.marker_colors[rid][0]
        m.color.g = self.marker_colors[rid][1]
        m.color.b = self.marker_colors[rid][2]
        # TODO constant values set at the top.
        m.scale.x = 0.5
        m.scale.y = 0.5
        # m.scale.x = 0.1
        p = self.graph.vs["coord"][vertex_id]
        p_t = self.latest_map.grid_to_pose(p)
        p_ros = Point(x=p_t[0], y=p_t[1])
        m.pose.position = p_ros
        self.marker_pub.publish(m)

    def publish_visited_vertices(self):
        rid = str(self.robot_id)
        # Publish visited vertices
        m = Marker()
        m.id = 2
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.POINTS
        # TODO constant values set at the top.
        m.color.a = 0.5
        m.color.b = self.marker_colors[rid][2]
        m.color.g = self.marker_colors[rid][1]
        m.color.r = self.marker_colors[rid][0]
        m.scale.x = 0.5
        m.scale.y = 0.5
        for vertex_id in self.visited_vertices:
            # Grid coordinate for the vertex.
            p = self.graph.vs["coord"][vertex_id]
            p_t = self.latest_map.grid_to_pose(p)
            p_ros = Point(x=p_t[0], y=p_t[1])
            m.points.append(p_ros)
        self.marker_pub.publish(m)

    def publish_gate_vertices(self, gate_poses):
        # Publish visited vertices
        gate = self.localize_gate(gate_poses)
        rid = str(self.robot_id)
        m = Marker()
        m.id = 2
        # if test:
        #     m.id = 20
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.POINTS
        # TODO constant values set at the top.
        m.color.a = 1.0
        m.color.b = self.marker_colors[rid][2]
        m.color.g = self.marker_colors[rid][1]
        m.color.r = self.marker_colors[rid][0]
        m.scale.x = 1.5
        m.scale.y = 1.5
        for vertex_id in [gate[-1]]:
            # Grid coordinate for the vertex.
            p = self.graph.vs["coord"][vertex_id]
            p_t = self.latest_map.grid_to_pose(p)
            p_ros = Point(x=p_t[0], y=p_t[1])
            m.points.append(p_ros)
        self.gate_marker_pub.publish(m)

    def publish_leaves(self):
        # Publish leaves
        if self.leaves:
            m = Marker()
            m.id = 3
            m.header.frame_id = self.latest_map.header.frame_id
            m.type = Marker.POINTS
            # TODO constant values set at the top.
            m.color.a = 1.0
            m.color.r = 1.0
            m.color.b = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            for vertex_id in self.leaves:
                # Grid coordinate for the vertex.
                p = self.graph.vs["coord"][vertex_id]
                p_t = self.latest_map.grid_to_pose(p)
                p_ros = Point(x=p_t[0], y=p_t[1])
                m.points.append(p_ros)
            self.marker_pub.publish(m)

    def publish_line(self, p, q):
        m = Marker()
        m.id = 7
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.LINE_STRIP
        # TODO constant values set at the top.
        m.color.a = 1.0
        m.color.r = 1.0
        m.scale.x = 0.1
        p_t = self.latest_map.grid_to_pose(p)
        p_ros = Point(x=p_t[0], y=p_t[1])
        m.points.append(p_ros)
        q_t = self.latest_map.grid_to_pose(q)
        q_ros = Point(x=q_t[0], y=q_t[1])
        m.points.append(q_ros)
        self.marker_pub.publish(m)

    def generate_graph(self):
        self.lock.acquire()
        start_time = time.clock()
        try:
            self.latest_map = Grid(self.get_map().map)
        except rospy.ServiceException:
            self.lock.release()
            pu.log_msg(self.robot_id, "Map didn't update", self.debug_mode)
            return
        self.min_range_radius = self.lidar_scan_radius / self.latest_map.resolution  # TODO range.
        self.min_edge_length = self.robot_radius / self.latest_map.resolution
        self.compute_gvg()
        # clean graph TODO remove after testing or you may not
        self.lock.release()
        now = time.clock()
        t = now - start_time
        self.performance_data.append(
            {'time': rospy.Time.now().to_sec(), 'type': 0, 'robot_id': self.robot_id, 'computational_time': t})

    def get_closest_vertex(self, robot_pose):
        """Get the closest vertex to robot_pose (x,y) in frame_id."""
        robot_grid = self.latest_map.pose_to_grid(robot_pose)
        vertices = np.asarray(self.graph.vs["coord"])

        return pu.get_closest_point(robot_grid, vertices)

    def find_shortest_path(self, s, graph, gate=[]):
        """
        Run DFS search from node s to t without traversing nodes in the gate
        :param s:
        :param t:
        :param gate:
        :return path from s to t:
        """

        def _remove_leaf(l, par):
            p = l
            while par[p]:
                neis = graph.neighbors(p, mode="all")
                if len(neis) == 1:
                    eid = graph.get_eid(par[p], p)
                    graph.delete_edges(eid)
                p = par[p]

        leaves_to_remove = []
        Q = [s]
        parent = {s: None}
        visited = []
        while len(Q) > 0:
            u = Q.pop()
            neis = graph.neighbors(u, mode="all")
            if len(neis) == 1:
                neighbor = graph.vs["coord"][neis[0]]
                current_vertex = graph.vs["coord"][u]
                if not self.latest_map.is_frontier(neighbor, current_vertex, self.min_range_radius):
                    leaves_to_remove.append(u)
            for v in neis:
                if v not in visited:
                    Q.append(v)
                    parent[v] = u
            visited.append(u)

        for l in leaves_to_remove:
            _remove_leaf(l, parent)

    def get_graph_path_to_leaf(self, robot_pose, leaf_pose):
        """
        Find the path (on the GVG) from a given pose in the world to a leaf. This defines the initial gate
        :param robot_pose:
        :param leaf_pose:
        :return (array of poses representing the gate):
        """
        gate_vids = []
        source_id, distance_source = self.get_closest_vertex(robot_pose)
        first_leaf_id, distance_leaf = self.get_closest_vertex(leaf_pose)
        gate_vids = \
            self.graph.get_shortest_paths(source_id, to=first_leaf_id, weights=self.graph.es["weight"],
                                          mode=igraph.ALL)[0]
        gate_poses = [self.latest_map.grid_to_pose(self.graph.vs["coord"][vid]) for vid in gate_vids]
        return gate_poses

    def localize_gate(self, gate_poses):
        """
        Given the previous gate poses, localize them on the new graph
        :param gate_poses : array of (x,y):
        :return vertex_ids for the all nodes on the gate: array of (x,y)
        """
        gate_vids = []
        source_id, distance_source = self.get_closest_vertex(gate_poses[0])
        first_leaf_id, distance_leaf = self.get_closest_vertex(gate_poses[-1])
        gate_vids = \
            self.graph.get_shortest_paths(source_id, to=first_leaf_id, weights=self.graph.es["weight"],
                                          mode=igraph.ALL)[0]
        return gate_vids

    def leaf_in_assigned_region(self, k, gate):
        """
        Check if a given leaf lies in the region defined by a given gate
        :param k vertex_id (int):
        :param gate (array of vertex_ids):
        :return boolean:
        """
        s = gate[0]
        t = gate[-1]
        psk = set(self.graph.get_shortest_paths(s, to=k, weights=self.graph.es["weight"], mode=igraph.ALL)[0])
        ptk = set(self.graph.get_shortest_paths(t, to=k, weights=self.graph.es["weight"], mode=igraph.ALL)[0])
        gate_set = set(gate)
        in_region = False
        in_region = len(gate_set.intersection(ptk)) == 1 and len(ptk) < len(psk)
        # else:
        #     m=ptk.intersection(psk)
        #     in_region=len(m.intersection(gate))==0
        return in_region, len(ptk)

    def get_new_gate(self, frontier_pose, old_gate):
        """
        Find the closest gate to current robot pose. Leaves to every gate are saved at the start of exploratino
        :param frontier_pose:
        :param old_gate:
        :return new_gate (array of poses):
        """
        source, dvertext = self.get_closest_vertex(old_gate[0])
        gate_leaf, dvertext = self.get_closest_vertex(frontier_pose)
        new_gate = \
            self.graph.get_shortest_paths(source, to=gate_leaf, weights=self.graph.es["weight"], mode=igraph.ALL)[0]
        new_gate_poses = [self.latest_map.grid_to_pose(self.graph.vs["coord"][vid]) for vid in new_gate]
        return new_gate_poses

    def get_paths_to_remaining_leaves(self, robot_vertex_id):
        non_visited_leaves = []
        valid_non_visited_leaves = []
        for p in self.other_leaves:
            vid, _ = self.get_closest_vertex(p)
            path_to_leaf = self.graph.get_shortest_paths(robot_vertex_id, to=vid, weights=self.graph.es["weight"],
                                                         mode=igraph.ALL)
            if len(path_to_leaf) >= 2 and self.latest_map.is_frontier(self.graph.vs["coord"][path_to_leaf[-2]],
                                                                      self.graph.vs["coord"][path_to_leaf[-1]],
                                                                      self.min_range_radius):
                non_visited_leaves.append(vid)
                valid_non_visited_leaves.append(p)
        paths_to_leaves = self.graph.get_shortest_paths(robot_vertex_id, to=non_visited_leaves,
                                                        weights=self.graph.es["weight"], mode=igraph.ALL)
        del self.other_leaves[:]
        self.other_leaves += valid_non_visited_leaves
        return paths_to_leaves

    def create_new_gate(self, initial_pose, current_pose, previous_poses, other_leaves, flagged_gate_leaves):
        """ Find the next region to explore using the gate nodes"""
        if not len(other_leaves):
            pu.log_msg(self.robot_id, "No gates left", 1 - self.debug_mode)
            return [], [], [], -1

        self.generate_graph()
        self.lock.acquire()

        # If there's any gate left, go ahead and pick one whose leaf is closest
        new_gate = []
        goal_grid = []
        prev_goal_grid = []
        gate_id = -1
        closest_path = []
        min_gate_len = np.inf
        prev_poses = []
        for lid, lpose in other_leaves.items():
            if lid not in flagged_gate_leaves:
                gate = self.get_graph_path_to_leaf(initial_pose, lpose)
                prev_poses = [gate[0], gate[-1]]
                self.lock.release()
                path_to_leaf = self.get_graph_path_to_leaf(current_pose, gate[-1])
                self.lock.acquire()
                if path_to_leaf:
                    path_len = len(path_to_leaf)
                    if path_len < min_gate_len:
                        gate_id = lid
                        new_gate = gate
                        min_gate_len = path_len
                        closest_path = path_to_leaf
        if gate_id != -1:
            goal_grid = self.graph.vs["coord"][self.get_closest_vertex(closest_path[-1])[0]]
            prev_goal_grid = self.graph.vs["coord"][self.get_closest_vertex(closest_path[-2])[0]]
            del previous_poses[:]  # clear previous history so that you don't revisit the previous region
            prev_poses += [closest_path[-2], closest_path[-1]]
            previous_poses += prev_poses
        else:
            pu.log_msg(self.robot_id, "No valid gate", 1 - self.debug_mode)
        self.lock.release()
        return new_gate, goal_grid, prev_goal_grid, gate_id

    def explored_gate(self, init_pose, leaf_pose):
        init_vid, _ = self.get_closest_vertex(init_pose)
        leaf_vid, _ = self.get_closest_vertex(leaf_pose)
        gate_vids = \
            self.graph.get_shortest_paths(init_vid, to=leaf_vid, weights=self.graph.es["weight"], mode=igraph.ALL)[0]
        parent_cell = self.graph.vs["coord"][gate_vids[-2]]
        leaf_cell = self.graph.vs["coord"][gate_vids[-1]]
        return self.latest_map.is_frontier(parent_cell, leaf_cell, self.min_range_radius)

    def get_successors(self, robot_pose, previous_pose=None, gate_poses=[], get_any=False):
        self.lock.acquire()
        start_time = time.clock()
        # Get current vertex.
        self.current_vertex_id, distance_current_vertex = self.get_closest_vertex(robot_pose)
        # self.publish_current_vertex(self.current_vertex_id)

        # find visited vertices
        self.visited_vertices = set()
        previous_previous_vertex_id = -1
        previous_vertex_id = -1
        for p in previous_pose:
            p_vertex_id, distance_p = self.get_closest_vertex(p)
            if distance_p < self.min_range_radius and p_vertex_id != self.current_vertex_id:  # TODO parameter.
                self.visited_vertices.add(p_vertex_id)
                previous_previous_vertex_id = previous_vertex_id
                previous_vertex_id = p_vertex_id
                if len(self.visited_vertices) > 1:
                    path_prev_current = self.graph.get_shortest_paths(previous_previous_vertex_id,
                                                                      to=previous_vertex_id,
                                                                      weights=self.graph.es["weight"], mode=igraph.ALL)
                    self.visited_vertices.update(path_prev_current[0])

        if previous_vertex_id != -1:
            path_prev_current = self.graph.get_shortest_paths(previous_vertex_id,
                                                              to=self.current_vertex_id,
                                                              weights=self.graph.es["weight"],
                                                              mode=igraph.ALL)
            self.visited_vertices.update(path_prev_current[0][:-1])
        # self.publish_visited_vertices()

        # find next node
        non_visited_leaves = []
        for l_vertex_id in self.leaves:
            if l_vertex_id not in self.visited_vertices and l_vertex_id != self.current_vertex_id:
                non_visited_leaves.append(l_vertex_id)

        # If current vertex is a leaf, then continue in that direction.
        paths_to_all_leaves = []
        if self.current_vertex_id in non_visited_leaves:
            paths_to_all_leaves += [self.latest_map.grid_to_pose(self.graph.vs["coord"][self.current_vertex_id])]

        # Find other leaves to visit
        paths_to_all_leaves = self.graph.get_shortest_paths(self.current_vertex_id,
                                                            to=non_visited_leaves, weights=self.graph.es["weight"],
                                                            mode=igraph.ALL)

        # Pick out only those in assigned region TODO @Kizito
        paths_to_leaves = []
        # self.publish_visited_vertices()
        gate = self.localize_gate(gate_poses)
        for path in paths_to_all_leaves:
            in_region, _ = self.leaf_in_assigned_region(path[-1], gate)
            if in_region:
                paths_to_leaves.append(path)

        if not get_any and not paths_to_leaves:
            pu.log_msg(self.robot_id,
                       "No paths. checking for leaves among those on stack: {}".format(self.other_leaves),
                       self.debug_mode)
            paths_to_leaves = self.get_paths_to_remaining_leaves(self.current_vertex_id)

        # after getting candidate paths, find the closest one
        full_path = []
        next_leaf = -1
        if paths_to_leaves:
            path_costs = [0] * len(paths_to_leaves)
            depth = 1
            path_to_leaf = -1
            path_length = np.inf
            max_depth = max([len(path) for path in paths_to_leaves])
            while depth < max_depth:
                counter = 0
                for i, path in enumerate(paths_to_leaves):
                    if depth < len(path):
                        if path_to_leaf != i:
                            path_costs[i] += self.graph.es["weight"][self.graph.get_eid(path[depth - 1], path[depth])]
                            if path[depth] not in self.visited_vertices:
                                if (path_to_leaf == -1 or path_costs[i] < path_costs[path_to_leaf]):
                                    path_to_leaf = i
                    if path_to_leaf != -1 and path_costs[i] > path_costs[path_to_leaf]:
                        counter += 1

                if counter == len(paths_to_leaves):
                    break
                depth += 1
            if path_to_leaf == -1:
                path_to_leaf = np.argmin(path_costs)

            for v in paths_to_leaves[path_to_leaf]:
                full_path.append(self.latest_map.grid_to_pose(self.graph.vs["coord"][v]))
                next_leaf = v

            # # save other leaves @Kizito
            if not get_any:
                for i in range(len(paths_to_leaves)):
                    if i != path_to_leaf:
                        self.other_leaves.append((self.graph.vs["coord"][paths_to_leaves[i][-1]]))
        else:
            if not get_any:
                pu.log_msg(self.robot_id, "NO MORE LEAVES LEFT: {}".format(self.other_leaves), self.debug_mode)
        if next_leaf != -1:
            self.publish_current_vertex(next_leaf)
        self.lock.release()
        return full_path

    def find_similar_slope_vertices(self, vertex_id):
        similar_slope_vertices = {}
        self.find_similar_slope_vertices_helper(vertex_id, similar_slope_vertices)

        return np.array(similar_slope_vertices.values())

    def find_similar_slope_vertices_helper(self, current_vertex_id, similar_slope_vertices):
        similar_slope_vertices[current_vertex_id] = self.graph.vs["coord"][current_vertex_id]
        if self.graph.degree(current_vertex_id) == 2:
            p_id, r_id = self.graph.neighbors(current_vertex_id)

            p = self.graph.vs["coord"][p_id]
            current_vertex = self.graph.vs["coord"][current_vertex_id]
            r = self.graph.vs["coord"][r_id]

            slope_pc = pu.get_slope(p, current_vertex)
            slope_cr = pu.get_slope(p, current_vertex)
            if np.isclose(slope_pc,
                          slope_cr, rtol=0.001, atol=0.001, equal_nan=True) or \
                    np.isclose(slope_cr, slope_pc, rtol=0.001, atol=0.001,
                               equal_nan=True):  # TODO tune.

                if p_id not in similar_slope_vertices:
                    self.find_similar_slope_vertices_helper(p_id, similar_slope_vertices)
                if r_id not in similar_slope_vertices:
                    self.find_similar_slope_vertices_helper(r_id, similar_slope_vertices)
        elif self.graph.degree(current_vertex_id) == 1:
            p_id = self.graph.neighbors(current_vertex_id)
            if p_id[0] not in similar_slope_vertices:
                self.find_similar_slope_vertices_helper(p_id[0], similar_slope_vertices)

    def should_communicate(self, robot_pose):
        """Return True if should communicate; false otherwise."""
        self.lock.acquire()
        start_time = time.clock()

        # self.merge_similar_edges2()

        # Get current vertex of the robot.
        self.current_vertex_id, distance_current_vertex = self.get_closest_vertex(robot_pose)
        current_vertex_id = self.current_vertex_id
        robot_grid = self.graph.vs["coord"][current_vertex_id]

        # self.publish_current_vertex(current_vertex_id, marker_id=5)
        # If at a junction point, might be worth to communicate.  TODO uncomment this if you want to use degree criteria
        # if self.graph.degree(self.current_vertex_id) > 2:
        #     pu.log_msg(self.robot_id, "intersection", self.debug_mode)
        #     self.lock.release()
        #     return True
        # else:
        #     vertices = np.array(self.intersections.values())
        #
        #     if vertices.size != 0:
        #         closest_intersection_point, dist = pu.get_closest_point(robot_grid, vertices)
        #         if dist < self.target_distance:  # TODO parameter
        #             pu.log_msg(self.robot_id, "intersection", self.debug_mode)
        #             self.lock.release()
        #             return True

        # Points on the same line for current_vertex_id
        current_vertex_similar_slope_vertices = self.find_similar_slope_vertices(current_vertex_id)
        # rospy.logerr("current_vertex_similar_slope_vertices {}".format(len(current_vertex_similar_slope_vertices)))
        # If can connect with another end, might be worth to communicate.
        for l_vertex_id in self.leaves:
            if l_vertex_id != current_vertex_id:
                rospy.sleep(0.01)
                # self.publish_current_vertex(l_vertex_id, marker_id=6) # TODO disable

                if pu.euclidean_distance(self.graph.vs["coord"][current_vertex_id], self.graph.vs["coord"][
                    l_vertex_id]) < self.comm_range / self.latest_map.resolution:  # TODO parameter for communication range
                    l_vertex_similar_slope_vertices = self.find_similar_slope_vertices(l_vertex_id)

                    try:
                        line_regress = pu.get_line(
                            np.vstack((current_vertex_similar_slope_vertices, l_vertex_similar_slope_vertices)))
                    except ValueError:
                        continue

                    if line_regress[2] < 2.0 and self.latest_map.line_in_unknown(
                            self.graph.vs["coord"][current_vertex_id],
                            self.graph.vs["coord"][l_vertex_id]):  # TODO parameter:
                        # rospy.logerr("line rmse {} {}".format(line_regress[2], np.vstack((current_vertex_similar_slope_vertices, l_vertex_similar_slope_vertices))))
                        p = [self.graph.vs["coord"][current_vertex_id][0],
                             line_regress[1] + line_regress[0] * self.graph.vs["coord"][current_vertex_id][0]]
                        q = [self.graph.vs["coord"][l_vertex_id][0],
                             line_regress[1] + line_regress[0] * self.graph.vs["coord"][l_vertex_id][0]]
                        # self.publish_line(p, q)  # TODO disable
                        self.lock.release()
                        return True
        end_time = time.clock()
        t = (end_time - start_time)
        pu.log_msg(self.robot_id, "should_communicate {}".format(t), self.debug_mode)
        self.performance_data.append(
            {'time': rospy.Time.now().to_sec(), 'type': 1, 'robot_id': self.robot_id,
             'computational_time': t})  # TODO not done if true
        self.lock.release()
        return False

    #####

    def graph_distance_handler(self, request):
        self.generate_graph()
        source = request.source
        targets = request.targets
        source_id, _ = self.get_closest_vertex((source.position.x, source.position.y))
        target_ids = []
        for p in targets:
            vid, _ = self.get_closest_vertex((p.position.x, p.position.y))
            target_ids.append(vid)
        paths_to_leaves = self.graph.get_shortest_paths(source_id, to=target_ids,
                                                        weights=self.graph.es["weight"], mode=igraph.ALL)
        dists = [len(path) for path in paths_to_leaves]
        return GraphDistanceResponse(distances=dists)

    def frontier_point_handler(self, request):
        pu.log_msg(self.robot_id, "received a request: {}".format(request.count), self.debug_mode)
        count = request.count
        current_goal_paths = request.current_paths
        self.generate_graph()

        start_time = time.clock()
        if len(self.leaves):
            best_leaf = max(self.leaves, key=self.leaves.get)
            selected_leaves = [best_leaf]
            scount = 0
            lcount = len(self.leaves)
            while scount <= lcount:  # UPDATE: Returning all leaves instead of just a few
                # if len(selected_leaves) == len(self.leaves):
                #     break
                for lid, area in self.leaves.items():
                    if lid not in selected_leaves:
                        v_dict = {}
                        for slid in selected_leaves:
                            v_count = self.count_path_vertices(lid, slid)
                            v_dict[v_count] = lid
                if v_dict:
                    max_val = max(list(v_dict))
                    max_lid = v_dict[max_val]
                    selected_leaves.append(max_lid)
                scount += 1

        frontiers = []
        coords = []
        for leaf_id in selected_leaves:
            p = self.latest_map.grid_to_pose(self.graph.vs["coord"][leaf_id])
            pcoord = (p[0], p[1])
            if pcoord not in coords:
                coords.append(pcoord)
                p_ros = Pose()
                p_ros.position.x = p[0]
                p_ros.position.y = p[1]
                frontiers.append(p_ros)

        self.fc_count += 1
        rospy.logerr("New FC count: {}".format(self.fc_count))
        pu.log_msg(self.robot_id, 'COMPUTED FRONTIER RESULTS: {}'.format(frontiers), self.debug_mode)
        return FrontierPointResponse(frontiers=frontiers)

    def get_leaf_path(self, p1, p2):
        v1 = self.get_closest_vertex(p1)
        v2 = self.get_closest_vertex((p2))
        p12 = self.graph.shortest_paths_dijkstra(source=v1[0], target=v2[0])
        return p12[0][0]

    def get_frontier_leaf_info(self):
        info = {}
        lcoord = {}
        for v in self.graph.vs:
            current_vertex_id = v.index
            if self.graph.degree(current_vertex_id) == 1:
                neighbor_id = self.graph.neighbors(current_vertex_id)[0]
                neighbor = self.graph.vs["coord"][neighbor_id]
                current_vertex = self.graph.vs["coord"][current_vertex_id]
                if not self.latest_map.is_frontier(neighbor, current_vertex, self.min_range_radius):
                    info[current_vertex_id] = self.latest_map.get_newinfo(neighbor, current_vertex,
                                                                          self.min_range_radius)
                    lcoord[current_vertex_id] = current_vertex
        return info, lcoord

    def compute_new_goals(self, current_paths):
        robot_vids = {rid: self.get_closest_vertex((p[0].position.x, p[0].position.y))[0] for rid, p in
                      current_paths.items()}
        leaf_info, leaf_coord = self.get_frontier_leaf_info()
        leaves = list(leaf_info)
        weights = {rid: {} for rid, vid in robot_vids.items()}
        for lv1 in leaves:
            obs = 0
            for lv2 in leaves:
                if lv1 != lv2:
                    obs += self.latest_map.get_path_obstacles(leaf_coord[lv1], leaf_coord[lv2])
            for r, rvid in robot_vids.items():
                plen = self.graph.shortest_paths_dijkstra(source=rvid, target=lv1)
                w = obs * leaf_info[lv1] / (plen[0][0] + 1)
                weights[r][w] = lv1

        next_goals = {}
        chosen_leaves = []
        goal_coordinates = [Pose()] * (max(list(robot_vids)) + 1)
        for rid, all_leaf_weights in weights.items():
            count = len(all_leaf_weights)
            while rid not in next_goals and count >= 0:
                bw = max(list(all_leaf_weights))
                best_leaf = all_leaf_weights[bw]
                if best_leaf not in chosen_leaves:
                    next_goals[rid] = best_leaf
                    grid_coord = leaf_coord[best_leaf]
                    pose = self.latest_map.grid_to_pose(grid_coord)
                    goal_coordinates[rid].position.x = pose[0]
                    goal_coordinates[rid].position.y = pose[1]
                    chosen_leaves.append(best_leaf)
                    break
                count -= 1
                del all_leaf_weights[bw]

        return goal_coordinates

    def are_valid_paths(self, goal_paths):
        goal_coords = {rid: self.graph.vs["coord"][self.get_closest_vertex((p[1].position.x, p[1].position.y))[0]] for
                       rid, p in goal_paths.items()}
        obstacle_exists_all = [True]
        for rid1, gc1 in goal_coords.items():
            for rid2, gc2 in goal_coords.items():
                if rid1 != rid2:
                    if self.latest_map.get_path_obstacles(gc1,
                                                          gc2) == 0:  # pu.D(self.latest_map.grid_to_pose(gc1),self.latest_map.grid_to_pose(gc2))<=self.lidar_scan_radius or
                        obstacle_exists_all.append(False)
                    else:
                        obstacle_exists_all.append(True)
        return all(obstacle_exists_all)

    def fetch_explored_region_handler(self, data):
        '''
        Map Analyzer use this to fetch explored cells
        :param data:
        :return:
        '''
        poses = self.latest_map.get_explored_region()
        return ExploredRegionResponse(poses=poses, resolution=self.map_resolution)

    ###### TO CHECK what is needed.

    def map_callback(self, map_msg):
        """Callback for Occupancy Grid."""
        pu.log_msg(self.robot_id, "Received map message", self.debug_mode)
        start_time_clock = time.clock()
        # Create a 2D grid.
        self.latest_map = Grid(map_msg)
        # Adjust min distance between obstacles in cells.
        self.min_edge_length = self.robot_radius / self.latest_map.resolution
        end_time_clock = time.clock()
        pu.log_msg(self.robot_id, "generate obstacles1 {}".format(end_time_clock - start_time_clock), self.debug_mode)
        # just for testing
        self.generate_graph()
        pu.log_msg(self.robot_id, "should comm {}".format(self.should_communicate(self.get_robot_pose())),
                   self.debug_mode)
        """
        if not self.plot_data_active:
            self.plot_data([], is_initial=True)
            rospy.logerr('Plotting complete')
        """

    def spin(self):
        r = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            try:
                # if not self.edges:
                #    self.generate_graph()
                r.sleep()
            except Exception as e:
                pu.log_msg(self.robot_id, 'Robot {}: Graph node interrupted!: {}'.format(self.robot_id, e),
                           self.debug_mode)

    def is_same_intersection(self, intersec, robot_pose):
        is_same = True
        if self.last_intersection:
            last_close_edge = self.last_intersection[0]
            current_close_edge = intersec[0]
            last_dist = min([pu.D(robot_pose, last_close_edge[0]), pu.D(robot_pose, last_close_edge[1])])
            current_dist = min([pu.D(robot_pose, current_close_edge[0]), pu.D(robot_pose, current_close_edge[1])])
            if abs(current_dist - last_dist) > self.comm_range:
                is_same = False
        return is_same

    def fetch_rendezvous_points_handler(self, data):
        count = data.count
        rendezvous_points = []
        all_points = self.latest_map.get_explored_region()
        origin = self.get_robot_pose()
        free_points = [p for p in all_points if pu.D(origin, p) <= self.comm_range]
        if free_points:
            hull, boundary_points = graham_scan(free_points, count, False)
            for ap in boundary_points:
                pose = Pose()
                pose.position.x = ap[INDEX_FOR_X]
                pose.position.y = ap[INDEX_FOR_Y]
                rendezvous_points.append(pose)
        return RendezvousPointsResponse(poses=rendezvous_points)

    def get_path(self, leaf, parent, parents, tree):
        l = leaf
        while parents[l] != parent:
            l = parents[l]
            tree.add(l)
        tree.add(l)

    def is_free(self, p):
        xc = int(np.round(p[INDEX_FOR_X]))
        yc = int(np.round(p[INDEX_FOR_Y]))
        new_p = [0] * 2
        new_p[INDEX_FOR_X] = xc
        new_p[INDEX_FOR_Y] = yc
        new_p = tuple(new_p)
        return new_p in self.pixel_desc and self.pixel_desc[new_p] == FREE

    def flood_fill(self, cells, prev_point, new_point, visited, frontiers):
        if new_point not in cells or new_point in visited:
            return
        if prev_point and cells[new_point] != cells[prev_point]:
            if cells[new_point] == UNKNOWN:
                frontiers[prev_point] = new_point
            return
        north_p = list(new_point)
        east_p = list(new_point)
        west_p = list(new_point)
        south_p = list(new_point)
        north_p[INDEX_FOR_X] += 1
        east_p[INDEX_FOR_X] -= 1
        south_p[INDEX_FOR_Y] += 1
        west_p[INDEX_FOR_Y] -= 1
        prev_point = new_point
        visited.append(new_point)

        self.flood_fill(cells, prev_point, tuple(north_p), visited, frontiers)
        self.flood_fill(cells, prev_point, tuple(east_p), visited, frontiers)
        self.flood_fill(cells, prev_point, tuple(south_p), visited, frontiers)
        self.flood_fill(cells, prev_point, tuple(west_p), visited, frontiers)

    def filter_obstacles(self, rpose, scan_poses):
        N = scan_poses.shape[0]
        distances = -1 * np.ones(N)
        for i in range(N):
            distances[i] = round(pu.T(rpose, scan_poses[i, :]), 6)
        pose_tally = np.asarray(np.unique(distances, return_counts=True)).T
        equidist_distances = pose_tally[pose_tally[:, 1] > 1]
        equidistant_pairs = []
        if equidist_distances.shape[0]:
            scan_poses[np.where(pose_tally == equidist_distances[i, 0]), :]
            equidist_distances += [scan_poses[np.where(pose_tally == equidist_distances[i, 0]), :] for i in
                                   range(equidist_distances.shape[0]) if equidist_distances[i, 0] != -1]

    def augment_graph(self, new_graph, vid1, vid2):
        """
        Add new subgraph to the existing graph
        :param new_graph:
        :param scan_pose:
        :return:
        """

        # # Get nodes from both graphs, which are closest to the current pose
        # vid1=self.get_closest_vertex_on_graph(self.current_pose,self.augmented_graph)
        # vid2=self.get_closest_vertex_on_graph(self.current_pose,new_graph)

        # Create an edge between the two nodes such that the two subgraphs are connected.
        bridge_id = len(self.augmented_graph.vs)
        self.augmented_graph.add_vertex(coord=new_graph.vs['coord'][vid2], name=bridge_id,
                                        pose=new_graph.vs['pose'][vid2])
        self.augmented_graph.add_edge(vid1, bridge_id, weight=euclidean_distance(self.augmented_graph.vs['coord'][vid1],
                                                                                 new_graph.vs['coord'][vid2]))

        # Add all edges on the new graph into the main graph
        new_vertex_id = bridge_id
        old_new_ids = {vid2: bridge_id}
        for edge in new_graph.es:
            sid = edge.source
            tid = edge.target
            new_sid = -1
            new_tid = -1
            if sid in old_new_ids:
                new_sid = old_new_ids[sid]
            if tid in old_new_ids:
                new_tid = old_new_ids[tid]
            if new_sid == -1:
                new_vertex_id += 1
                new_sid = new_vertex_id
                old_new_ids[sid] = new_sid
                self.augmented_graph.add_vertex(coord=new_graph.vs['coord'][sid], name=new_sid,
                                                pose=new_graph.vs['pose'][sid])
            if new_tid == -1:
                new_vertex_id += 1
                new_tid = new_vertex_id
                old_new_ids[tid] = new_tid
                self.augmented_graph.add_vertex(coord=new_graph.vs['coord'][tid], name=new_tid,
                                                pose=new_graph.vs['pose'][tid])
            self.augmented_graph.add_edge(new_sid, new_tid, weight=euclidean_distance(new_graph.vs['coord'][sid],
                                                                                      new_graph.vs['coord'][tid]))
        rospy.logerr("old_graph: {}, subgraph: {}, new graph: {}".format(bridge_id, len(new_graph.vs), new_vertex_id))

        # self.publish_border_edge([vid1,bridge_id],self.augmented_graph)
        # self.publish_scan_edges(vid1,bridge_id,self.augmented_graph)

    def get_closest_vertex_on_graph(self, robot_pose, graph):
        """
        Get the closest vertex to robot_pose (x,y) in frame_id from a given graph
        :param robot_pose:
        :param graph:
        :return vertex id:
        """
        robot_grid = self.latest_map.pose_to_grid(robot_pose)
        vertices = np.asarray(graph.vs["coord"])
        vertex_id, _ = pu.get_closest_point(robot_grid, vertices)
        return vertex_id

    def generate_gvg(self, obstacles, robot_pose):
        """
        Generates a voronoi diagram that is induced by given obstables
        :param obstacles:
        :return voronoi graph:
        """
        start_id = 0  # set starting vertex id for new graph so that the new graph has unique ids
        if self.augmented_graph:
            start_id = max(self.augmented_graph.vs['name']) + 1

        vor = Voronoi(obstacles)

        # fig,ax=plt.subplots(figsize=(18,18))
        # voronoi_plot_2d(vor,ax=ax)
        # plt.savefig("/home/masaba/Winter21/voronoi_figures/robot{}_{}.png".format(self.robot_id,rospy.Time.now().to_sec()))

        # Initializing the graph.
        online_graph = igraph.Graph()
        # Correspondance between graph vertex and Voronoi vertex IDs.
        gvg_correspondance = {}

        # Simplifying access of Voronoi data structures.
        vertices = vor.vertices
        ridge_vertices = vor.ridge_vertices
        ridge_points = vor.ridge_points
        # Create a graph based on ridges.
        for i in xrange(len(ridge_vertices)):
            ridge_vertex = ridge_vertices[i]
            # If any of the ridge vertices go to infinity, then don't add.
            if ridge_vertex[0] == -1 or ridge_vertex[1] == -1:
                continue
            p1 = vertices[ridge_vertex[0]]
            p2 = vertices[ridge_vertex[1]]

            # Obstacle points determining the ridge.
            ridge_point = ridge_points[i]
            q1 = obstacles[ridge_point[0]]
            q2 = obstacles[ridge_point[1]]

            # If the vertices on the ridge are in the free space
            # and distance between obstacle points is large enough for the robot
            if self.latest_map.is_free(p1[INDEX_FOR_X], p1[INDEX_FOR_Y]) and \
                    self.latest_map.is_free(p2[INDEX_FOR_X], p2[INDEX_FOR_Y]) and \
                    euclidean_distance(q1, q2) > self.min_edge_length:

                # Add vertex and edge.
                graph_vertex_ids = [-1, -1]  # temporary for finding verted IDs.

                # Determining graph vertex ID if existing or not.
                for point_id in xrange(len(graph_vertex_ids)):
                    if ridge_vertex[point_id] not in gvg_correspondance:
                        # if not existing, add new vertex.
                        graph_vertex_ids[point_id] = online_graph.vcount()
                        p_t = self.latest_map.grid_to_pose(vertices[ridge_vertex[point_id]])
                        p_ros = (p_t[0], p_t[1])
                        online_graph.add_vertex(coord=vertices[ridge_vertex[point_id]],
                                                name=graph_vertex_ids[point_id] + start_id, pose=p_ros)

                        gvg_correspondance[ridge_vertex[point_id]] = graph_vertex_ids[point_id]
                    else:
                        # Otherwise, already added before.
                        graph_vertex_ids[point_id] = gvg_correspondance[ridge_vertex[point_id]]
                # Add edge.
                online_graph.add_edge(graph_vertex_ids[0], graph_vertex_ids[1],
                                      weight=euclidean_distance(p1, p2))
            else:
                # Otherwise, edge not added.
                continue

        # Take only the largest component.
        cl = online_graph.clusters()
        online_graph = cl.giant()

        # Find node that is closest to the current robot pose.
        grid_pose = robot_pose[0:2]
        yaw = robot_pose[-1]
        pose_dist = {vid: euclidean_distance(grid_pose, online_graph.vs['coord'][vid]) for vid in
                     range(online_graph.vcount())}
        vid2 = min(pose_dist, key=pose_dist.get)
        vid1 = None
        if self.augmented_graph and len(self.augmented_graph.vs):
            aug_pose_dist = {vid: euclidean_distance(grid_pose, self.augmented_graph.vs['coord'][vid]) for vid in
                             range(self.augmented_graph.vcount())}
            vid1 = min(aug_pose_dist, key=aug_pose_dist.get)
        return online_graph, vid1, vid2

    def publish_obstacles(self, scan_poses):
        # Publish visited vertices
        m = Marker()
        m.id = 2
        m.header.frame_id = "robot_{}/map".format(self.robot_id)
        m.type = Marker.POINTS
        # TODO constant values set at the top.
        m.color.a = 1.0
        m.color.g = 0.5
        if self.robot_id == 0:
            m.color.b = 0.5
        elif self.robot_id == 1:
            m.color.g = 0.5
        m.scale.x = 0.2
        m.scale.y = 0.2
        for pose in scan_poses:
            p_ros = Point(x=pose[0], y=pose[1])
            m.points.append(p_ros)
        self.obstacle_pose_pub.publish(m)

    def publish_scan_edges(self, s, graph):
        """For debug, publishing of GVG."""
        # Marker that will contain line sequences.
        rid = str(self.robot_id)
        m = Marker()
        m.id = 0
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.LINE_LIST
        m.color.a = 1.0
        m.scale.x = 0.1
        m.color.r = self.marker_colors[rid][0]
        m.color.g = self.marker_colors[rid][1]
        m.color.b = self.marker_colors[rid][2]
        self.find_shortest_path(s, graph)
        for edge in graph.get_edgelist():
            for vertex_id in edge:
                p_t = graph.vs["pose"][vertex_id]
                p_ros = Point(x=p_t[0], y=p_t[1])
                m.points.append(p_ros)
        self.scan_marker_pub.publish(m)

    def publish_border_edge(self, vertex_id, graph):
        """For debug, publishing of GVG."""
        # Marker that will contain line sequences.
        rid = str(self.robot_id)
        m = Marker()
        m.id = 0
        m.header.frame_id = self.latest_map.header.frame_id
        m.type = Marker.POINTS
        m.color.a = 1.0
        m.scale.x = 1.0
        m.color.r = self.marker_colors[rid][0]
        m.color.g = self.marker_colors[rid][1]
        m.color.b = self.marker_colors[rid][2]
        p_t = graph.vs["coord"][vertex_id]
        p_ros = Point(x=p_t[0], y=p_t[1])
        m.points.append(p_ros)
        self.border_edge_pub.publish(m)

    def save_all_data(self, data):
        save_data(self.performance_data,
                  "{}/performance_{}_{}_{}_{}_{}_{}.pickle".format(self.method, self.environment, self.robot_count,
                                                                   self.run,
                                                                   self.termination_metric, self.robot_id,
                                                                   self.max_target_info_ratio))

    def get_robot_pose(self):
        # TODO add in a different class.
        robot_pose = None
        while not robot_pose:
            if True:
                self.tf_listener.waitForTransform("robot_{}/map".format(self.robot_id),
                                                  "robot_{}/base_link".format(self.robot_id),
                                                  rospy.Time(0),
                                                  rospy.Duration(4.0))
                (robot_loc_val, rot) = self.tf_listener.lookupTransform("robot_{}/map".format(self.robot_id),
                                                                        "robot_{}/base_link".format(self.robot_id),
                                                                        rospy.Time(0))
                robot_pose = robot_loc_val[0:2]
            # except:
            #    rospy.sleep(1)
            #    pass
        robot_pose = np.array(robot_pose)
        return robot_pose


if __name__ == "__main__":
    rospy.init_node("gvg_graph_node")

    graph = Graph()
    rospy.Subscriber('/robot_{}/map'.format(graph.robot_id),
                     OccupancyGrid, graph.map_callback)
    graph.spin()
