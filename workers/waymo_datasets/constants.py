import tensorflow as tf


_num_map_samples = 30000
A_MIN = 1e-6
A_MAX = 1000000000


ROADGRAPH_FEATURES = {
    'roadgraph_samples/dir': tf.io.FixedLenFeature(
        [_num_map_samples, 3], tf.float32, default_value=None
    ),
    'roadgraph_samples/id': tf.io.FixedLenFeature(
        [_num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/type': tf.io.FixedLenFeature(
        [_num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/valid': tf.io.FixedLenFeature(
        [_num_map_samples, 1], tf.int64, default_value=None
    ),
    'roadgraph_samples/xyz': tf.io.FixedLenFeature(
        [_num_map_samples, 3], tf.float32, default_value=None
    ),
}
# Features of other agents.
STATE_FEATURES = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
}

TRAFFIC_LIGHT_FEATURES = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

FEATURES_DESCRIPTION = {}
FEATURES_DESCRIPTION.update(ROADGRAPH_FEATURES)
FEATURES_DESCRIPTION.update(STATE_FEATURES)
FEATURES_DESCRIPTION.update(TRAFFIC_LIGHT_FEATURES)

OBJECT_TYPE = {
    0: 'TYPE_UNSET',
    1: 'TYPE_VEHICLE',
    2: 'TYPE_PEDESTRIAN',
    3: 'TYPE_CYCLIST',
    4: 'TYPE_OTHER'
}

LANE_TYPE = {
    0: 'TYPE_UNDEFINED',
    1: 'TYPE_FREEWAY',
    2: 'TYPE_SURFACE_STREET',
    3: 'TYPE_BIKE_LANE'
}

ROAD_LINE_TYPE = {
    0: 'TYPE_UNKNOWN',
    1: 'TYPE_BROKEN_SINGLE_WHITE',
    2: 'TYPE_SOLID_SINGLE_WHITE',
    3: 'TYPE_SOLID_DOUBLE_WHITE',
    4: 'TYPE_BROKEN_SINGLE_YELLOW',
    5: 'TYPE_BROKEN_DOUBLE_YELLOW',
    6: 'TYPE_SOLID_SINGLE_YELLOW',
    7: 'TYPE_SOLID_DOUBLE_YELLOW',
    8: 'TYPE_PASSING_DOUBLE_YELLOW'
}

ROAD_EDGE_TYPE = {
    0: 'TYPE_UNKNOWN',
    # // Physical road boundary that doesn't have traffic on the other side (e.g.,
    # // a curb or the k-rail on the right side of a freeway).
    1: 'TYPE_ROAD_EDGE_BOUNDARY',
    # // Physical road boundary that separates the car from other traffic
    # // (e.g. a k-rail or an island).
    2: 'TYPE_ROAD_EDGE_MEDIAN'
}

POLYLINE_TYPE = {
    # for lane
    'TYPE_UNDEFINED': -1,
    'TYPE_FREEWAY': 1,
    'TYPE_SURFACE_STREET': 2,
    'TYPE_BIKE_LANE': 3,
    # for roadline
    'TYPE_UNKNOWN': -1,
    'TYPE_BROKEN_SINGLE_WHITE': 6,
    'TYPE_SOLID_SINGLE_WHITE': 7,
    'TYPE_SOLID_DOUBLE_WHITE': 8,
    'TYPE_BROKEN_SINGLE_YELLOW': 9,
    'TYPE_BROKEN_DOUBLE_YELLOW': 10,
    'TYPE_SOLID_SINGLE_YELLOW': 11,
    'TYPE_SOLID_DOUBLE_YELLOW': 12,
    'TYPE_PASSING_DOUBLE_YELLOW': 13,
    # for roadedge
    'TYPE_ROAD_EDGE_BOUNDARY': 15,
    'TYPE_ROAD_EDGE_MEDIAN': 16,
    # for stopsign
    'TYPE_STOP_SIGN': 17,
    # for crosswalk
    'TYPE_CROSSWALK': 18,
    # for speed bump
    'TYPE_SPEED_BUMP': 19
}

SIGNAL_STATE = {
    0: 'LANE_STATE_UNKNOWN',
    # // States for traffic signals with arrows.
    1: 'LANE_STATE_ARROW_STOP',
    2: 'LANE_STATE_ARROW_CAUTION',
    3: 'LANE_STATE_ARROW_GO',
    # // Standard round traffic signals.
    4: 'LANE_STATE_STOP',
    5: 'LANE_STATE_CAUTION',
    6: 'LANE_STATE_GO',
    # // Flashing light signals.
    7: 'LANE_STATE_FLASHING_STOP',
    8: 'LANE_STATE_FLASHING_CAUTION'
}

SIGNAL_STATE_TO_ID = {v: k for k, v in SIGNAL_STATE.items()}