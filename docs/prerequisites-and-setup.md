# ⚒️ Prerequisites and Setup

## 🔧 Prerequisites

Refer to the [Carla simulator document](https://carla.readthedocs.io/en/stable/) to get to know about the Carla simulator. Install [docker](https://docs.docker.com/) and have it running. Once you are done with this, you may clone the [SDC](https://github.com/NikhilKamathB/SDC) repository. Follow the **Setup** section to have the Carla simulator up and running.

## 🧰 Setup

* Install Nvidia Drivers
* For this project, we will have the Carla server up via the docker. To run Carla on docker, refer [this](https://carla.readthedocs.io/en/latest/build\_docker/) doc. Remember to install Nvidia container toolkit.

```
docker pull carlasim/carla:<tag>      # eg: carlasim/carla:0.9.15 

docker run --rm -it -d --privileged --gpus all --net=host --name carla-server carlasim/carla:<tag> /bin/bash ./CarlaUE4.sh -RenderOffScreen
```

* Alternatively, if you want to have the server up with rendering enabled, you may follow these steps:

```
xhost +local:docker      # Allow docker containers to display on the host's X server

docker run --rm -it -d --privileged --gpus all --net=host --name carla-server -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix carlasim/carla:<tag> /bin/bash ./CarlaUE4.sh
```

* Create and activate an environment. Note that this project is using `Python 3.8`. Install requirements using pip from the requirements file found in `./SDC` folder:

```
pip install -r requirements.txt
pip install -r requirements-simpan.txt    # [Optional] - use this with SimPan
```

* If you running the container with rendering enabled - remember to revert the permissions you set with xhost for security reasons once you're done - `xhost -local:docker`.
* Note that if you want to run the python client (getting pygame window up in Mac) from your Mac, then you have to setup X11 Forwarding. Do the follwing for setting this up:
  * Download and install XQuartz from [here](https://www.xquartz.org/).
  * After installation, log out and log back in to ensure XQuartz starts properly.
  * Upon opening the XQuartz terimal, run `ssh -Y username@remote-server` for secure connection. You may then simply run your script.
  * The downside is that the visualization and communication will have a noticable latency.
* Another note - if you are using conda environment remember to run `conda install -c conda-forge libstdcxx-ng` wiith your environment active (to solve MESA-LOADER error).&#x20;

## 🤫 Environment Variables

```
export HOSTNAME="localhost"
export PORT="2000"
```

## 🧱 Folder Structure

```
SDC Repo.

|- backup (they just contain the backup data/assets)
|- data (any data, structured/unstructured, goes here - taken care by DVC)
    |- assets (any static and long-lived objects goes here - will be public)
    |- config (all the generated/custom configurations for the actors in the environment must be defeined here)
    |- raw (holds unprocessed information)
    |- processed (holds processed information)
|- logs (logging of information will be done here; logging must be in the following format `log_<timestamp>.<extension>`)
|- src (driving code goes)
    |- base (holds the definition of handlers like actors, ego vehivles, maps, etc)
    |- model (contains pydantic and enums models that define attributes for the enviroment)
    |- uitls (untilities are defined here)
    |- client.py (establistes the Carla client and acts as a gateway between our custom scripts and the Carla server)
    |- data_synthesizer.py (script for generating and storing any kind of data from any actor's point of view)
|- test (all test cases are defined here)
|- main.py (contains the driver code)
|- requirements.txt (holds dependencies)
|- requirements-simpan.txt (holds dependencies - when this repo will be used as submodule with SimPam)
```
