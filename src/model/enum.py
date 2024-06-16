##########################################################################################################
# Store all the enums representing the different types of entities in the project here
##########################################################################################################

import carla
from enum import Enum


class DistanceMetric(Enum):
    
    """
    Define the different distance metrics.
    """
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class SearchAlgorithm(Enum):
        
    """
    Define the different search algorithms.
    """
    BREADTH_FIRST_SEARCH = "bfs"
    DEPTH_FIRST_SEARCH = "dfs"
    UNIFORM_COST_SEARCH = "ucs"
    A_STAR = "astar"


class SensorConvertorType(Enum):

    """
    Define the different types of sensor convertors.
    """
    RGB = "rgb"
    DEPTH = "depth"
    LOGARITHMIC_DEPTH = "logarithmic_depth"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"


class SpectatorAttachmentMode(Enum):

    """
    Define spectator atttachment mode.
    """
    
    DEFAULT = "d"
    VEHICLE = "v"
    PEDESTRIAN = "p"


class TMActorSpeedMode(Enum):
    
    """
    Define the different speed modes for the traffic manager actors in km/h.
    """
    DEFAULT = 60.0
    SLOW = 15.0
    NORMAL = 45.0
    FAST = 80.0
    MIN = 0.0
    MAX = 100.0
    RANDOM = -1.0


class VehicleDoor(Enum):
    """
    Define the different doors of a vehicle.
    """
    ALL = carla.VehicleDoor.All
    FRONT_LEFT = carla.VehicleDoor.FL
    FRONT_RIGHT = carla.VehicleDoor.FR
    REAR_LEFT = carla.VehicleDoor.RL
    REAR_RIGHT = carla.VehicleDoor.RR


class VehicleLightState(Enum):
    """
    Define the different lights of a vehicle.
    """
    ALL = carla.VehicleLightState.All
    NONE = carla.VehicleLightState.NONE
    POSITION = carla.VehicleLightState.Position
    LOW_BEAM = carla.VehicleLightState.LowBeam
    HIGH_BEAM = carla.VehicleLightState.HighBeam
    BRAKE = carla.VehicleLightState.Brake
    RIGHT_BLINKER = carla.VehicleLightState.RightBlinker
    LEFT_BLINKER = carla.VehicleLightState.LeftBlinker
    REVERSE = carla.VehicleLightState.Reverse
    FOG = carla.VehicleLightState.Fog
    INTERIOR = carla.VehicleLightState.Interior
    SPECIAL_1 = carla.VehicleLightState.Special1
    SPECIAL_2 = carla.VehicleLightState.Special2


class Gen1VehicleType(Enum):

    """
    Define the different types of generation 1 vehicles that can be spawned in the environment.
    """

    AUDI_A2 = "vehicle.audi.a2"
    AUDI_ETRON = "vehicle.audi.etron"
    AUDI_TT = "vehicle.audi.tt"
    BMW_GRAND_TOURER = "vehicle.bmw.grandtourer"
    CHEVROLET_IMPALA = "vehicle.chevrolet.impala"
    CITROEN_C3 = "vehicle.citroen.c3"
    DOGE_CHARGER_POLICE = "vehicle.dodge.charger_police"
    FORD_MUSTANG = "vehicle.ford.mustang"
    JEEP_WRANGLER_RUBICON = "vehicle.jeep.wrangler_rubicon"
    LINCOLN_MKZ_2017 = "vehicle.lincoln.mkz_2017"
    MERCEDES_COUPE = "vehicle.mercedes.coupe"
    MICRO_MICROLINO = "vehicle.micro.microlino"
    MINI_COOPER_S = "vehicle.mini.cooper_s"
    NISSAN_MICRA = "vehicle.nissan.micra"
    NISSAN_PATROL = "vehicle.nissan.patrol"
    SEAT_LEON = "vehicle.seat.leon"
    TESLA_MODEL3 = "vehicle.tesla.model3"
    TOYOTA_PRIUS = "vehicle.toyota.prius"
    CARLA_MOTORS_CARLA_COLA = "vehicle.carlamotors.carlacola"
    VOLKSWAGEN_T2 = "vehicle.volkswagen.t2"
    HARLEY_DAVIDSON_LOW_RIDER = "vehicle.harley-davidson.low_rider"
    KAWASAKI_NINJA = "vehicle.kawasaki.ninja"
    VESPA_ZX_125 = "vehicle.vespa.zx125"
    YAMAHA_YZF = "vehicle.yamaha.yzf"
    BH_CROSSBIKE = "vehicle.bh.crossbike"
    DIAMONDBACK_CENTURY = "vehicle.diamondback.century"
    GAZELLE_OMAFIETS = "vehicle.gazelle.omafiets"


class Gen2VehicleType(Enum):

    """
    Define the different types of generation 2 vehicles that can be spawned in the environment.
    """

    DOGE_CHARGER_2020 = "vehicle.dodge.charger_2020"
    DOGE_CHARGER_POLICE_2020 = "vehicle.dodge.charger_police_2020"
    FORD_CROWN_TAXI = "vehicle.ford.crown"
    LINCOLN_MKZ_2020 = "vehicle.lincoln.mkz_2020"
    MERCEDES_COUPE_2020 = "vehicle.mercedes.coupe_2020"
    MINI_COOPER_S_2021 = "vehicle.mini.cooper_s_2021"
    NISSAN_PATROL_2021 = "vehicle.nissan.patrol_2021"
    CARLA_MOTORS_EUROPEAN_HGV = "vehicle.carlamotors.european_hgv"
    CARLA_MOTORS_FIRETRUCK = "vehicle.carlamotors.firetruck"
    TESLA_CYBERTRUCK = "vehicle.tesla.cybertruck"
    FORD_AMBULANCE = "vehicle.ford.ambulance"
    MERCEDES_SPRINTER = "vehicle.mercedes.sprinter"
    VOLKSWAGEN_T2_2021 = "vehicle.volkswagen.t2_2021"
    MISTUBISHI_FUSOROSA = "vehicle.mitsubishi.fusorosa"


class WalkerType(Enum):
    """
    Define the different types of walkers that can be spawned in the environment.
    """
    PEDESTRIAN_1_1 = "walker.pedestrian.0001"
    PEDESTRIAN_1_2 = "walker.pedestrian.0005"
    PEDESTRIAN_1_3 = "walker.pedestrian.0006"
    PEDESTRIAN_1_4 = "walker.pedestrian.0007"
    PEDESTRIAN_1_5 = "walker.pedestrian.0008"
    PEDESTRIAN_2_1 = "walker.pedestrian.0004"
    PEDESTRIAN_2_2 = "walker.pedestrian.0003"
    PEDESTRIAN_2_3 = "walker.pedestrian.0002"
    PEDESTRIAN_3_1 = "walker.pedestrian.0015"
    PEDESTRIAN_3_2 = "walker.pedestrian.0019"
    PEDESTRIAN_4_1 = "walker.pedestrian.0016"
    PEDESTRIAN_4_2 = "walker.pedestrian.0017"
    PEDESTRIAN_5_1 = "walker.pedestrian.0026"
    PEDESTRIAN_5_2 = "walker.pedestrian.0018"
    PEDESTRIAN_6_1 = "walker.pedestrian.0021"
    PEDESTRIAN_6_2 = "walker.pedestrian.0020"
    PEDESTRIAN_7_1 = "walker.pedestrian.0023"
    PEDESTRIAN_7_2 = "walker.pedestrian.0022"
    PEDESTRIAN_8_1 = "walker.pedestrian.0024"
    PEDESTRIAN_8_2 = "walker.pedestrian.0025"
    PEDESTRIAN_9_1 = "walker.pedestrian.0027"
    PEDESTRIAN_9_2 = "walker.pedestrian.0029"
    PEDESTRIAN_9_3 = "walker.pedestrian.0028"
    PEDESTRIAN_10_1 = "walker.pedestrian.0041"
    PEDESTRIAN_10_2 = "walker.pedestrian.0040"
    PEDESTRIAN_10_3 = "walker.pedestrian.0033"
    PEDESTRIAN_10_4 = "walker.pedestrian.0031"
    PEDESTRIAN_11_1 = "walker.pedestrian.0034"
    PEDESTRIAN_11_2 = "walker.pedestrian.0038"
    PEDESTRIAN_12_1 = "walker.pedestrian.0035"
    PEDESTRIAN_12_2 = "walker.pedestrian.0036"
    PEDESTRIAN_12_3 = "walker.pedestrian.0037"
    PEDESTRIAN_13_1 = "walker.pedestrian.0039"
    PEDESTRIAN_14_1 = "walker.pedestrian.0042"
    PEDESTRIAN_14_2 = "walker.pedestrian.0043"
    PEDESTRIAN_14_3 = "walker.pedestrian.0044"
    PEDESTRIAN_15_1 = "walker.pedestrian.0047"
    PEDESTRIAN_15_2 = "walker.pedestrian.0046"
    PEDESTRIAN_C_1_1 = "walker.pedestrian.0011"
    PEDESTRIAN_C_1_2 = "walker.pedestrian.0010"
    PEDESTRIAN_C_1_3 = "walker.pedestrian.0009"
    PEDESTRIAN_C_2_1 = "walker.pedestrian.0014"
    PEDESTRIAN_C_2_2 = "walker.pedestrian.0013"
    PEDESTRIAN_C_2_3 = "walker.pedestrian.0012"
    PEDESTRIAN_C_3_1 = "walker.pedestrian.0048"
    PEDESTRIAN_C_4_1 = "walker.pedestrian.0049"
    PEDESTRIAN_P_1 = "walker.pedestrian.0030"
    PEDESTRIAN_P_2 = "walker.pedestrian.0032"