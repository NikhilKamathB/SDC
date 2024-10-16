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

# ⚒️ Prerequisites and Setup

## 🔧 Prerequisites

Refer to the [Carla simulator document](https://carla.readthedocs.io/en/stable/) to get to know about the Carla simulator. Install [docker](https://docs.docker.com/) and have it running. Once you are done with this, you may clone the [SDC](https://github.com/NikhilKamathB/SDC) repository. Follow the **Setup** sections to have the the entire repo active and running. The **Setup 1** section involves manual installation of dependencies and packages. The next section **Setup 2** is pretty straight forward and must be very simple.

## 🧰 Setup 1

* Have accounts setup in GitHub and GitLab to access all modules required.
* Install Docker, `C++` Compiler and `CMake`.
* If you intend to have the Carla Simulator setup,  you will need the following:
  * Make sure you have Iinstalled Nvidia Drivers
  * For this project, we will have the Carla server up via the docker. To run Carla on docker, refer [this](https://carla.readthedocs.io/en/latest/build\_docker/) doc. Remember to install [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  * <pre><code><strong>docker pull carlasim/carla:&#x3C;tag>      # eg: carlasim/carla:0.9.15 
    </strong>
    docker run --rm -it -d --privileged --gpus all --net=host --name carla-server carlasim/carla:&#x3C;tag> /bin/bash ./CarlaUE4.sh -RenderOffScreen
    </code></pre>
* Have a python environment created and activate it.&#x20;
* In this project we will be making use of Argoverse datasets and their utilities. Refer the [Agroverse guide](https://argoverse.github.io/user-guide/argoverse\_2.html) for a deeper dive. As per their instructions ([downloading the ave package](https://argoverse.github.io/user-guide/getting\_started.html#installing-via-pip)), we will be install Rust. You may install Rust from [here](https://www.rust-lang.org/tools/install). Once you have installed Rust, switch to the nightly build of Rust as per the [Agroverse installation guide](https://argoverse.github.io/user-guide/getting\_started.html#installing-via-pip) by running `rustup default nightly`.&#x20;

## 💼 Setup 2

This step has to be executed using the `./setup.sh` script that can be found at the root of the parent directory. The bash script does the following:

1. Clearing logs
2. Updates the DVC components
3. Installs the Argoverse dependencies
4. Installs required Python packages
5. Update the git submodules.
6. Builds the \`[Algorithms](https://github.com/NikhilKamathB/Algorithms)\` library.
7. Have services up and running using docker compose.

* For this to work you have to set the following environment variables:
  * `CLEAR_LOGS = <bool` - enables log clearing
  * `UPDATE_DVC_CONFIG = <bool>` - enables DVC config update
  * `DVC_DATA_PATH = <str>` - Path to DVC config
  * `AV2_DIRECTORY = <str>` - Agroverse editable/package path
  * `WAYMO_CELERY_BASE_URL = <str>` - Celery base URL for Waymo worker
  * `WAYMO_CELERY_DATABASE_NUMBER = <str>` - Celery database number for Waymo worker
  * `WAYMO_CELERY_LOG_DIR = <str>` - Celery logging directory path
* With your environment activated, run

```
./setup.sh
```

## 💱Miscellaneous

* If you want to have the carla server up with rendering enabled, you may follow these steps:

```
xhost +local:docker      # Allow docker containers to display on the host's X server

docker run --rm -it -d --privileged --gpus all --net=host --name carla-server -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix carlasim/carla:<tag> /bin/bash ./CarlaUE4.sh
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
export CLEAR_LOGS=<BOOL>
export UPDATE_DVC_CONFIG=<BOOL>
export DVC_DATA_PATH="<PATH_TO_DVC_LOCAL_STORAGE>"
export INSTALL_SIMPAN_DEPENDENCIES=<BOOL>
export AV2_DIRECTORY="<PATH_TO_STORE_AGROVERSE_PROJECT>"
export WAYMO_CELERY_BASE_URL="<WAYMO_CELERY_BROKER_URL>"
export WAYMO_CELERY_DATABASE_NUMBER="<DB_NUMBER>"
export WAYMO_CELERY_LOG_DIR="<WAYMO_CELERY_LOGGING_DIR>"
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
    |- agroverse (contains agroverse specefic code)
    |- waymo (container waymo specefic code)
    |- base (holds the definition of handlers like actors, ego vehivles, maps, etc)
    |- model (contains pydantic and enums models that define attributes for the enviroment)
    |- uitls (untilities are defined here)
    |- client.py (establistes the Carla client and acts as a gateway between our custom scripts and the Carla server)
    |- data_synthesizer.py (script for generating and storing any kind of data from any actor's point of view)
    |- motion_planning.py (script for motionl planning for our ego vehicle)
|- test (all test cases are defined here)
|- third_party (contains git submodules)
|- workers (contains workers for executing isolated tasks)
|- main.py (contains the driver code)
|- utils.py (utilities for the `main.py` file)
|- setup.sh (bash script to setup the project)
|- requirements.txt (holds dependencies)
|- requirements-simpan.txt (holds dependencies - when this repo will be used as submodule with SimPam)
```
