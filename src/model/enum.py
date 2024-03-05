##########################################################################################################
# Store all the enums representing the different types of entities in the project here
##########################################################################################################

import carla
from enum import Enum
    

class VehicleDoor(Enum):

    """
    Define the different doors of a vehicle.
    """

    All = carla.VehicleDoor.All
    FrontLeft = carla.VehicleDoor.FL
    FrontRight = carla.VehicleDoor.FR
    RearLeft = carla.VehicleDoor.RL
    RearRight = carla.VehicleDoor.RR


class VehicleLightState(Enum):

    """
    Define the different lights of a vehicle.
    """

    All = carla.VehicleLightState.All
    NoneLight = carla.VehicleLightState.NONE
    Position = carla.VehicleLightState.Position
    LowBeam = carla.VehicleLightState.LowBeam
    HighBeam = carla.VehicleLightState.HighBeam
    Brake = carla.VehicleLightState.Brake
    RightBlinker = carla.VehicleLightState.RightBlinker
    LeftBlinker = carla.VehicleLightState.LeftBlinker
    Reverse = carla.VehicleLightState.Reverse
    Fog = carla.VehicleLightState.Fog
    Interior = carla.VehicleLightState.Interior
    Special1 = carla.VehicleLightState.Special1
    Special2 = carla.VehicleLightState.Special2


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

    pedestrian_1_1 = "walker.pedestrian.0001"
    pedestrian_1_2 = "walker.pedestrian.0005"
    pedestrian_1_3 = "walker.pedestrian.0006"
    pedestrian_1_4 = "walker.pedestrian.0007"
    pedestrian_1_5 = "walker.pedestrian.0008"
    pedestrian_2_1 = "walker.pedestrian.0004"
    pedestrian_2_2 = "walker.pedestrian.0003"
    pedestrian_2_3 = "walker.pedestrian.0002"
    pedestrian_3_1 = "walker.pedestrian.0015"
    pedestrian_3_2 = "walker.pedestrian.0019"
    pedestrian_4_1 = "walker.pedestrian.0016"
    pedestrian_4_2 = "walker.pedestrian.0017"
    pedestrian_5_1 = "walker.pedestrian.0026"
    pedestrian_5_2 = "walker.pedestrian.0018"
    pedestrian_6_1 = "walker.pedestrian.0021"
    pedestrian_6_2 = "walker.pedestrian.0020"
    pedestrian_7_1 = "walker.pedestrian.0023"
    pedestrian_7_2 = "walker.pedestrian.0022"
    pedestrian_8_1 = "walker.pedestrian.0024"
    pedestrian_8_2 = "walker.pedestrian.0025"
    pedestrian_9_1 = "walker.pedestrian.0027"
    pedestrian_9_2 = "walker.pedestrian.0029"
    pedestrian_9_3 = "walker.pedestrian.0028"
    pedestrian_10_1 = "walker.pedestrian.0041"
    pedestrian_10_2 = "walker.pedestrian.0040"
    pedestrian_10_3 = "walker.pedestrian.0033"
    pedestrian_10_4 = "walker.pedestrian.0031"
    pedestrian_11_1 = "walker.pedestrian.0034"
    pedestrian_11_2 = "walker.pedestrian.0038"
    pedestrian_12_1 = "walker.pedestrian.0035"
    pedestrian_12_2 = "walker.pedestrian.0036"
    pedestrian_12_3 = "walker.pedestrian.0037"
    pedestrian_13_1 = "walker.pedestrian.0039"
    pedestrian_14_1 = "walker.pedestrian.0042"
    pedestrian_14_2 = "walker.pedestrian.0043"
    pedestrian_14_3 = "walker.pedestrian.0044"
    pedestrian_15_1 = "walker.pedestrian.0047"
    pedestrian_15_2 = "walker.pedestrian.0046"
    pedestrian_c_1_1 = "walker.pedestrian.0011"
    pedestrian_c_1_2 = "walker.pedestrian.0010"
    pedestrian_c_1_3 = "walker.pedestrian.0009"
    pedestrian_c_2_1 = "walker.pedestrian.0014"
    pedestrian_c_2_2 = "walker.pedestrian.0013"
    pedestrian_c_2_3 = "walker.pedestrian.0012"
    pedestrian_c_3_1 = "walker.pedestrian.0048"
    pedestrian_c_4_1 = "walker.pedestrian.0049"
    pedestrian_p_1 = "walker.pedestrian.0030"
    pedestrian_p_2 = "walker.pedestrian.0032"
