# Self-Driving Car.
Self-driving cars have been one of the most challenging and fascinating fields of study in automation. The main motive behind such a concept is to provide effortless and comfortable driving. The evolution of this discipline has pushed many researchers and enthusiasts to build automobiles that achieve level five autonomy. This repository aims to implement a simulated car with level four autonomy. Inspired by the timeline of self-driving cars and ever-growing technologies, the work has adopted the Carla simulator as a base for formulating the problem statement. It is open-source and offers various functionalities one can think about in relation to self-driving cars (SDC). This repository branches out to address concepts of visual perception, state estimation and localization, and motion planning. A previous version of this project can be found [here](https://drive.google.com/file/d/1IXyGhBM2OLqZS4HTRtfFpyoeYW-f11aI/view?usp=share_link). This version is part of a continuous development effort.

## Objective
Design a Level 4 Autonomous Vehicle and serve as a platform for another project **SimPan**.

## Prerequisites.
<p style="text-align: justify;">
&nbsp;&nbsp;&nbsp;&nbsp; Refer to the <a href="https://carla.readthedocs.io/en/stable/">Carla simulator document</a> to get to know about the Carla simulator. Install <a href="https://docs.docker.com">docker</a> and have it running. Once you are done with this, you may clone this repository. Follow the <strong>Setup</strong> section to have the Carla simulator up and running.
</p>

## Setup.
* Install Nvidia drivers.
<!-- * Refer [1](https://carla.readthedocs.io/en/latest/start_quickstart/) and [2](https://carla.readthedocs.io/en/latest/adv_rendering_options/), then run:
```
sudo apt-get install libxerces-c3.2 libomp5 xserver-xorg mesa-utils libvulkan1

sudo nvidia-xconfig --preserve-busid -a --virtual=1280x1024

export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

./CarlaUE4.sh -quality-level=low
``` -->
* For this project, we will have the Carla server up via the docker. To run Carla on docker, refer [this](https://carla.readthedocs.io/en/latest/build_docker/) doc. Remember to install Nvidia container toolkit.
```
docker pull carlasim/carla:<tag>      # eg: carlasim/carla:0.9.15 

docker run --rm -it -d --privileged --gpus all --net=host --name carla-server carlasim/carla:<tag> /bin/bash ./CarlaUE4.sh -RenderOffScreen
```
* Create and activate an environment. Note that this project is using `Python 3.8`. Install requirements using pip from the requirements file found in `./SDC` folder:
```
pip install -r requirements.txt
```

## Folder Structure
```
SDC Repo.

|- data (any data, structured/unstructured, goes here)
    |- raw (holds unprocessed information)
    |- processed (holds processed information)
|- logs (logging of information will be done here; logging must be in the following format `<timestamp>/log_<suffix>.<extension>`)
|- src (driving code goes)
    |- data (contains scripts to generate/process/store data)
|- config.json (any configuration, specific to the player, Carla server, etc, will be declared here)
|- main.py (contains the driver code)
|- data_generate.sh (bash script to open pygame and generate data)
|- requirements.txt (holds dependencies)
```

## Tasks.
- [ ] Data acquisition.
    - [ ] Automate data collection over pygame using different sensors
    - [ ] Code storing scripts
- [ ] High level motion planning
- [ ] State estimation and localization
- [ ] Low level motion planning
- [ ] Build perception stack

## Environment Variables
* GOOGLE_APPLICATION_CREDENTIALS