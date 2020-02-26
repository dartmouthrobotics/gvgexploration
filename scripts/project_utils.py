import math
import numpy as np
import pickle
from os import path

TOTAL_COVERAGE = 1
MAXIMUM_EXPLORATION_TIME = 2
COMMON_COVERAGE = 3
FULL_COVERAGE = 4

INDEX_FOR_X = 0
INDEX_FOR_Y = 1
SMALL = 0.00000001
PRECISION = 0
FREE = 0
OCCUPIED = 100
UNKNOWN = -1
SCALE = 10.0

# navigation states
ACTIVE_STATE = 1  # This state shows that the robot is collecting messages
PASSIVE_STATE = -1  # This state shows  that the robot is NOT collecting messages
ACTIVE = 1  # The goal is currently being processed by the action server
SUCCEEDED = 3  # The goal was achieved successfully by the action server (Terminal State)
ABORTED = 4  # The goal was aborted during execution by the action server due to some failure (Terminal State)
LOST = 9  # An action client can determine that a goal is LOST. This should not be sent over the wire by an action


def save_data(data, file_name):
    saved_data = []
    if not path.exists(file_name):
        f = open(file_name, "wb+")
        f.close()
    else:
        saved_data = load_data_from_file(file_name)
    saved_data += data
    with open(file_name, 'wb') as fp:
        pickle.dump(saved_data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        fp.close()


def load_data_from_file(file_name):
    data_dict = []
    if path.exists(file_name) and path.getsize(file_name) > 0:
        with open(file_name, 'rb') as fp:
            # try:
            data_dict = pickle.load(fp)
            fp.close()
            # except Exception as e:
            #     rospy.logerr("error saving data: {}".format(e))
    return data_dict


def pose2pixel(pose, origin_x, origin_y, resolution):
    x = round((pose[INDEX_FOR_X] - origin_y) / resolution, PRECISION)
    y = round((pose[INDEX_FOR_Y] - origin_x) / resolution, PRECISION)
    position = [0.0] * 2
    position[INDEX_FOR_X] = x
    position[INDEX_FOR_Y] = y
    return tuple(position)


def pixel2pose(point, origin_x, origin_y, resolution):
    new_p = [0.0] * 2
    new_p[INDEX_FOR_X] = origin_x + point[INDEX_FOR_X] * resolution
    new_p[INDEX_FOR_Y] = origin_y + point[INDEX_FOR_Y] * resolution
    return tuple(new_p)


def get_vector(p1, p2):
    xv = p2[INDEX_FOR_X] - p1[INDEX_FOR_X]
    yv = p2[INDEX_FOR_Y] - p1[INDEX_FOR_Y]
    v = [0] * 2
    v[INDEX_FOR_X] = xv
    v[INDEX_FOR_Y] = yv
    v = tuple(v)
    return v


def theta(p, q):
    dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
    dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
    return math.atan2(dy, dx)


def D(p, q):
    dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
    dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
    return math.sqrt(dx ** 2 + dy ** 2)


def T(p, q):
    return D(p, q) * math.cos(theta(p, q))


def W(p, q):
    return abs(D(p, q) * math.sin(theta(p, q)))


def slope(p, q):
    dx = q[INDEX_FOR_X] - p[INDEX_FOR_X]
    dy = q[INDEX_FOR_Y] - p[INDEX_FOR_Y]
    if dx == 0:
        dx = SMALL
        if dy < 0:
            return -1 / dx
        return 1 / dx
    return dy / dx


def get_vector(p1, p2):
    xv = p2[INDEX_FOR_X] - p1[INDEX_FOR_X]
    yv = p2[INDEX_FOR_Y] - p1[INDEX_FOR_Y]
    v = [0] * 2
    v[INDEX_FOR_X] = xv
    v[INDEX_FOR_Y] = yv
    v = tuple(v)
    return v


def get_ridge_desc(ridge):
    p1 = ridge[0][0]
    p2 = ridge[0][1]
    q1 = ridge[1][0]
    q2 = ridge[1][1]
    return p1, p2, q1, q2


def line_points(p1, p2, parts):
    x_min = int(round(min([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])))
    y_min = int(round(min([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])))
    x_max = int(round(max([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])))
    y_max = int(round(max([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])))
    pts = zip(np.linspace(x_min, x_max, parts), np.linspace(y_min, y_max, parts))
    points = []
    for p in pts:
        point = [0.0] * 2
        point[INDEX_FOR_X] = round(p[0], PRECISION)
        point[INDEX_FOR_Y] = round(p[1], PRECISION)
        points.append(tuple(point))
    return points


def compute_similarity(v1, v2, e1, e2):
    dv1 = np.sqrt(v1[INDEX_FOR_X] ** 2 + v1[INDEX_FOR_Y] ** 2)
    dv2 = np.sqrt(v2[INDEX_FOR_X] ** 2 + v2[INDEX_FOR_Y] ** 2)
    dotv1v2 = v1[INDEX_FOR_X] * v2[INDEX_FOR_X] + v1[INDEX_FOR_Y] * v2[INDEX_FOR_Y]
    v1v2 = dv1 * dv2
    if v1v2 == 0:
        v1v2 = 1
    if abs(dotv1v2) == 0:
        dotv1v2 = 0
    cos_theta = round(dotv1v2 / v1v2, PRECISION)
    sep = separation(e1, e2)
    return cos_theta, sep


def get_linear_points(intersections, lidar_fov):
    linear_ridges = []
    for intersect in intersections:
        p1 = intersect[0][0]
        p2 = intersect[0][1]
        p3 = intersect[1][1]
        if D(p2, p3) > lidar_fov and collinear(p1, p2, p3):
            linear_ridges.append(intersect)
    return linear_ridges


def collinear(p1, p2, p3, width, bias):
    s1 = slope(p1, p2)
    s2 = slope(p2, p3)
    if bias >= abs(s1 - s2) and 2 * W(p2, p3) <= width:
        return True
    return False


def scale_up(pose):
    p = [0.0] * 2
    p[INDEX_FOR_X] = round(pose[INDEX_FOR_X] * SCALE, PRECISION)
    p[INDEX_FOR_Y] = round(pose[INDEX_FOR_Y] * SCALE, PRECISION)
    p = tuple(p)
    return p


def scale_down(pose):
    p = [0.0] * 2
    p[INDEX_FOR_X] = pose[INDEX_FOR_X] / SCALE
    p[INDEX_FOR_Y] = pose[INDEX_FOR_Y] / SCALE
    p = tuple(p)
    return p


def process_edges(edges):
    x_pairs = []
    y_pairs = []
    edge_list = list(edges)
    for edge in edge_list:
        xh, yh = reject_outliers(list(edge))
        if len(xh) == 2:
            x_pairs.append(xh)
            y_pairs.append(yh)
    return x_pairs, y_pairs


def separation(e1, e2):
    p1 = e1[0]
    p2 = e1[1]
    p3 = e2[0]
    p4 = e2[1]
    c1 = p1[INDEX_FOR_Y] - slope(p1, p2) * p1[INDEX_FOR_X]
    c2 = p4[INDEX_FOR_Y] - slope(p3, p4) * p4[INDEX_FOR_X]
    return abs(c1 - c2)


def is_free(p, pixel_desc):
    rounded_pose = get_point(p)
    return rounded_pose in pixel_desc and pixel_desc[rounded_pose] == FREE


def is_obstacle(p, pixel_desc):
    new_p = get_point(p)
    return new_p in pixel_desc and pixel_desc[new_p] == OCCUPIED


def get_point(p):
    xc = round(p[INDEX_FOR_X], PRECISION)
    yc = round(p[INDEX_FOR_Y], PRECISION)
    new_p = [0.0] * 2
    new_p[INDEX_FOR_X] = xc
    new_p[INDEX_FOR_Y] = yc
    new_p = tuple(new_p)
    return new_p


def reject_outliers(data):
    raw_x = [v[INDEX_FOR_X] for v in data]
    raw_y = [v[INDEX_FOR_Y] for v in data]
    # rejected_points = [v for v in raw_x if v < 0]
    # indexes = [i for i in range(len(raw_x)) if raw_x[i] in rejected_points]
    # x_values = [raw_x[i] for i in range(len(raw_x)) if i not in indexes]
    # y_values = [raw_y[i] for i in range(len(raw_y)) if i not in indexes]
    # return x_values, y_values
    return raw_x, raw_y


def in_range(point, polygon):
    x = point[INDEX_FOR_X]
    y = point[INDEX_FOR_Y]
    return polygon[0][INDEX_FOR_X] <= x <= polygon[2][INDEX_FOR_X] and polygon[0][INDEX_FOR_Y] <= y <= polygon[2][
        INDEX_FOR_Y]


def create_polygon(pose, for_frontiers, origin_x, origin_y, width, height, comm_range):
    x = pose[INDEX_FOR_X]
    y = pose[INDEX_FOR_Y]
    first = [0] * 2
    second = [0] * 2
    third = [0] * 2
    fourth = [0] * 2
    #
    if for_frontiers:
        first[INDEX_FOR_Y] = origin_x
        first[INDEX_FOR_X] = origin_x

        second[INDEX_FOR_Y] = origin_x
        second[INDEX_FOR_X] = origin_y + height

        third[INDEX_FOR_Y] = origin_x + width
        third[INDEX_FOR_X] = origin_y + height

        fourth[INDEX_FOR_Y] = origin_x + width
        fourth[INDEX_FOR_X] = origin_y
    else:
        first[INDEX_FOR_Y] = x - comm_range
        first[INDEX_FOR_X] = y - comm_range

        second[INDEX_FOR_Y] = x - comm_range
        second[INDEX_FOR_X] = y + comm_range

        third[INDEX_FOR_Y] = x + comm_range
        third[INDEX_FOR_X] = y + comm_range

        fourth[INDEX_FOR_Y] = x + comm_range
        fourth[INDEX_FOR_X] = y - comm_range

    ranges = [first, second, third, fourth]
    return ranges


def there_is_unknown_region(p1, p2, pixel_desc, min_ratio=4.0):
    x_min = int(round(min([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])))
    y_min = int(round(min([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])))
    x_max = int(round(max([p1[INDEX_FOR_X], p2[INDEX_FOR_X]])))
    y_max = int(round(max([p1[INDEX_FOR_Y], p2[INDEX_FOR_Y]])))
    points = []
    point_count = 0
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            point_count += 1
            region_point = [0.0] * 2
            region_point[INDEX_FOR_X] = float(x)
            region_point[INDEX_FOR_Y] = float(y)
            region_point = tuple(region_point)
            if region_point in pixel_desc and pixel_desc[region_point] == UNKNOWN:
                points.append(region_point)
    return len(points) >= point_count / min_ratio
