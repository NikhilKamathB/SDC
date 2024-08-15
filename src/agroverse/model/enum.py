
##########################################################################################################
# Store all the enums for AV2 Dataset here
##########################################################################################################

from enum import Enum


class AV2ForecastingAnalyticsAttributes(Enum):
    """
        Define analytics attributes.
        """
    DIRECTORY = "directory"
    SCENARIO_ID = "scenario_id"
    N_PEDESTRIANS = "n_pedestrians"
    AVG_PEDESTRIAN_DISTANCE = "avg_pedestrian_distance"
    MEDIAN_PEDESTRIAN_DISTANCE = "median_pedestrian_distance"
    MAX_PEDESTRIAN_DISTANCE = "max_pedestrian_distance"
    MAX_PEDESTRIAN_DISTANCE_ID = "max_pedestrian_distance_id"
    MIN_PEDESTRIAN_DISTANCE = "min_pedestrian_distance"
    MIN_PEDESTRIAN_DISTANCE_ID = "min_pedestrian_distance_id"
    N_CYCLISTS = "n_cyclists"
    AVG_CYCLIST_DISTANCE = "avg_cyclist_distance"
    MEDIAN_CYCLIST_DISTANCE = "median_cyclist_distance"
    MAX_CYCLIST_DISTANCE = "max_cyclist_distance"
    MAX_CYCLIST_DISTANCE_ID = "max_cyclist_distance_id"
    MIN_CYCLIST_DISTANCE = "min_cyclist_distance"
    MIN_CYCLIST_DISTANCE_ID = "min_cyclist_distance_id"
    N_MOTORCYCLISTS = "n_motorcyclists"
    AVG_MOTORCYCLIST_DISTANCE = "avg_motorcyclist_distance"
    MEDIAN_MOTORCYCLIST_DISTANCE = "median_motorcyclist_distance"
    MAX_MOTORCYCLIST_DISTANCE = "max_motorcyclist_distance"
    MAX_MOTORCYCLIST_DISTANCE_ID = "max_motorcyclist_distance_id"
    MIN_MOTORCYCLIST_DISTANCE = "min_motorcyclist_distance"
    MIN_MOTORCYCLIST_DISTANCE_ID = "min_motorcyclist_distance_id"
    N_VEHICLES = "n_vehicles"
    AVG_VEHICLE_DISTANCE = "avg_vehicle_distance"
    MEDIAN_VEHICLE_DISTANCE = "median_vehicle_distance"
    MAX_VEHICLE_DISTANCE = "max_vehicle_distance"
    MAX_VEHICLE_DISTANCE_ID = "max_vehicle_distance_id"
    MIN_VEHICLE_DISTANCE = "min_vehicle_distance"
    MIN_VEHICLE_DISTANCE_ID = "min_vehicle_distance_id"
    N_BUSES = "n_buses"
    AVG_BUS_DISTANCE = "avg_bus_distance"
    MEDIAN_BUS_DISTANCE = "median_bus_distance"
    MAX_BUS_DISTANCE = "max_bus_distance"
    MAX_BUS_DISTANCE_ID = "max_bus_distance_id"
    MIN_BUS_DISTANCE = "min_bus_distance"
    MIN_BUS_DISTANCE_ID = "min_bus_distance_id"
    AV_DISTANCE = "av_distance"
