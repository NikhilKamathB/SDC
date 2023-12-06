# Self-Driving Car.
<p style="text-align: justify;">
&nbsp;&nbsp;&nbsp;&nbsp; Self-Driving cars have been one of the most challenging and fascinating fields of study in automation. The main motive of such a concept is to provide effortless and comfortable driving. Evolution of this discipline has pushed many researchers and enthusiasts in building automobiles to achieve level five autonomy. This repo aims at implementing a simulated car with level four autonomy. Inspired by the timeline of self-driving cars and ever-growing technologies, the work has taken up Carla simulator as a base for the problem statement formulation. It is open-source and provides various functionalities one can think about when it comes to SDC. This repo branches out to deal with concepts of visual perception, state estimation and localization and motion planning. 
</p>

## Before diving.
<p style="text-align: justify;">
&nbsp;&nbsp;&nbsp;&nbsp; Refer to the <a href="https://carla.readthedocs.io/en/stable/getting_started/">Carla simulator document</a> to setup the simulator on your system. Once the the setup is complete install the necessary libraries and place the cloned repository in <code>./CARLA_version>/PythonAPI/</code> directory.
</p>

## Setup.
* Install Nvidia drivers.
* Refer [1](https://carla.readthedocs.io/en/latest/start_quickstart/) and [2](https://carla.readthedocs.io/en/latest/adv_rendering_options/), then run:
```
sudo apt-get install libxerces-c3.2 libomp5 xserver-xorg mesa-utils libvulkan1

sudo nvidia-xconfig --preserve-busid -a --virtual=1280x1024

export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/nvidia_icd.json"

./CarlaUE4.sh -quality-level=low
```
* (Optional) To run Carla on docker, refer [this](https://carla.readthedocs.io/en/latest/build_docker/) doc. Remember to install Nvidia container toolkit. Run the server with RenderOffScreen flag set, refer [open issue](https://github.com/carla-simulator/carla/issues/4755).
```
docker pull carlasim/carla:0.9.12

docker run -d --privileged --gpus all --net=host carlasim/carla:0.9.12 /bin/bash ./CarlaUE4.sh -RenderOffScreen
```
* Install requirements using pip from the requirements file found in ```/CARLA_version/PythonAPI/SDC/``` folder:
```
pip install -r requirements.txt
```

## Methodology.
* [ ] In-progress.

## Directories and files.
<p style="text-align: justify;">
&nbsp;&nbsp;&nbsp;&nbsp; The repository has two main wings: the data generator and the simulation scripts. The control flows from the 'main.py' in the repository's root directory to either of these branched depending upon the command line arguments passed during execution. Files and folders are organized as follows:

* DATA
* LOGS
* MODELS
* DATA_SCRIPTS
* SIMULATION_SCRIPTS
* config.py
* main.py
* requirements.txt
</p>

## How to run?
## Results.
## References.
* [Capture data while manually controlling the agent.](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py)