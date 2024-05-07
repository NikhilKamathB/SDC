# 🏃 How do I use this project

The codebase is designed such that all commands needed for the execution of the client resides in `main.py` - aka the driver module. We establish the Carla client as a CLI application using which they can perform tasks such as data annotaion, automation, state estimation, OEDR, motion planning, basically any ascpect of the self-driving car.

## Generating actor configurations

Refer [this document](https://carla.readthedocs.io/en/latest/core\_actors/) to know the differnet types of actors provided by Carla.

* Make sure you have a base configuration for any actor defined in the designated directory. In our case, you can find `vehicle0.yaml`and `pedestrian0.yaml` in `./data/config/vehicles/` and `./data/config/pedestrians/` respectively - Experiment 0 😝.
* To get to know our configuration CLI command, run:

```
python -m main generate_configuration --help 
```

* To generate configurations for vehicles, you have to run (defaults to vehicle):

```
python -m main generate_vehicle_configuration
```

* To generate configurations for pedestrians, run:

```
python -m main generate_configuration --config-dir=<CONFIG_DIR> \
                                      --reference-config-file=<REFERENCE_YAML_FILE> \
                                      --for-pedestrian --no-for-vehicle 
```

## Generating synthetic data in Carla

Refer [this document](https://carla.readthedocs.io/en/latest/core\_sensors/) to know the differnet types of sensors provided by Carla.

* To get to know our data acquisition CLI command, run:

```
python -m main generate_synthetic_data --help
```

* You may set the command's arguments/options as per you need and run the command.

```
python -m main generate_synthetic_data [OPTIONS] 
```
