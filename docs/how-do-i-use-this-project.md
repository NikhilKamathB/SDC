---
layout:
  title:
    visible: true
  description:
    visible: false
  tableOfContents:
    visible: true
  outline:
    visible: true
  pagination:
    visible: true
---

# 🏃 How do I use this project

The codebase is designed such that all commands needed for the execution of the client resides in `main.py` - aka the driver module. We establish the Carla client as a CLI application using which they can perform tasks such as data annotaion, automation, state estimation, OEDR, motion planning, basically any ascpect of the self-driving car. The tool also supports subsidiaries that allows us to connect with other data providers like Agroverse, Waymo, etc. and leverage them to improve our pipeline. To learn more about the CLI run `python -m main --help`.

## Carla - Generating actor configurations

Refer [this document](https://carla.readthedocs.io/en/latest/core\_actors/) to know the differnet types of actors provided by Carla.

* Make sure you have a base configuration for any actor defined in the designated directory. In our case, you can find `vehicle0.yaml`and `pedestrian0.yaml` in `./data/config/vehicles/` and `./data/config/pedestrians/` respectively - Experiment 0 😝.
* To get to know our configuration CLI command, run:

```
python -m main carla generate_configuration --help 
```

* To generate configurations for vehicles, you have to run (defaults to vehicle):

```
python -m main carla generate_vehicle_configuration
```

* To generate configurations for pedestrians, run:

```
python -m main carla generate_configuration --config-dir=<CONFIG_DIR> \
                                      --reference-config-file=<REFERENCE_YAML_FILE> \
                                      --for-pedestrian --no-for-vehicle 
```

## Generating synthetic data in Carla

Refer [this document](https://carla.readthedocs.io/en/latest/core\_sensors/) to know the differnet types of sensors provided by Carla.

* To get to know our data acquisition CLI command, run the below mentioned code. You may use the relevant setting / default setting to run this CLI.

```
python -m main carla generate_synthetic_data --help    # To get information about this CLI

python -m main carla generate_synthetic_data [OPTIONS]    # To run the code
```

## Performing Motion Planning in Carla

The following are the references used to perform motion planning for an ego vehicle in the Carla simulator - [Algorithms](https://github.com/NikhilKamathB/Algorithms/blob/main/README.md).

* Before performing the motion planning make sure to build the "Algorithms" library as described in the above referenced doc. You may also refer to the [Prerequisites and Setup](prerequisites-and-setup.md) section to build the necessary modules.
* Motion planning CLI purpose is two fold - High Level Motion Planning and Low Level Motion Planning.
* To get to know more about the High Level Motion Planning, run the below mentioned code. You may use the relevant setting / default setting to run this CLI.

```
python -m main motion_planning generate_route --help.   # To get information about this CLI

python -m main motion_planning generate_route [OPTIONS]    # To run the code
```

## Agroverse Dataset

All tasks under this section are executed as Celery workers using docker. For implementation details check the `./workers/av2_datasets` directory. You can learn more about the associated CLI tool by running `python -m main av2_dataset --help`.

### AV2 - Motion Forecasting Dataset

* The AV2 motion forecasting dataset can be found [here](https://www.argoverse.org/av2.html#forecasting-link). You can run the following code to get to know more about the av2 motion forcasting CLI.

```
python -m main av2_dataset visualize_agroverse_data --help
```

* To use the default av2 visualization provided run

```
python -m main av2_dataset visualize_agroverse_data
```

* You can plot a detailed version of the scenario using

```
python -m main av2_dataset visualize_agroverse_data --no-raw --show-pedestrian-xing --scenario-id <SCENARIO_ID>
```

* The AV2 motion forecasting dataset CLI also comes with `generate_analytics_agroverse_forecasting_data` command. This command skims through the raw data and gets information specific to each scenarios - number of pedestrians, their average speed, number of vehicles, etc. To learn more, run

```
python -m main av2_dataset generate_analytics_agroverse_forecasting_data --help
```

* You may use the `query av2_forecasting_query_max_occurrence` command to get scenarios based on maximum occurrence of an entity. Run the following command to learn more

```
python -m main av2_dataset query --help

python -m main av2_dataset query av2_forecasting_query_max_occurrence --help
```

## Waymo Dataset

All tasks under this section are executed as Celery workers using docker. For implementation details check the `./workers/waymo_datasets` directory. You can learn more about the associated CLI tool by running `python -m main waymo_dataset --help`. For now, this cannot be run on Mac M1 and above chipset (not tested on windows, but it should work).

### Waymo Open Motion Dataset

* The Waymo open motion dataset can be found [here](https://waymo.com/open/). You can run the following code to get to know more about the waymo-open-motion dataset CLI.

```
python -m main waymo_dataset visualize_waymo_open_motion_data --help
```

* To use the default way open motion dataset visualizaition provided run

```
python -m main waymo_dataset visualize_waymo_open_motion_data
```

* We have also provided a way to convert tf-records into pickled/json files for every secnario within a record. This preprocessing step is motivated by the MTR paper implementation and the code is taken from [here](https://github.com/sshaoshuai/MTR/blob/master/mtr/datasets/waymo/data\_preprocess.py). You can learn more about this tool by simply running

```
python -m main waymo_dataset preprocess_waymo_open_motion_data --help
```

## Running Workers Independently

* This projects contains a folder `./workers`. Any task that has to be run in an isolated fashion is defined here. Each worker is executed as a celery job. Setting workers/jobs are taken in the setup script, so there is nothing to be done explicitly.
* Supposed you want deeper control over these individual workers (while developing, you might have to alter code, test, etc), you can do so by running these workers separately. Every worker has a `Dockerfile` and a `docker-compose.yml` file. `cd` into the corresponding worker directory and build the image  - `docker build -t sdc-av2-dataset .` - and start containers by running `docker compose up`. You can use "Dev Containers" extension in vscode to attach the container continue working as usual.

## Knowing how the workers are doing

* When the `./setup.sh` script is executed, it creates `redis-insight` and `flower` containers. You can open [http://localhost:5540](http://localhost:5540) and [http://localhost:5555](http://localhost:5555) to know the status of your jobs.
