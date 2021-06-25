# Self-Driving Car.
<p style="text-align: justify;">
&nbsp;&nbsp;&nbsp;&nbsp; Self-Driving cars have been one of the most challenging and fascinating fields of study in automation. The main motive of such a concept is to provide effortless and comfortable driving. Evolution of this discipline has pushed many researchers and enthusiasts in building automobiles to achieve level five autonomy. This repo aims at implementing a simulated car with level three autonomy which could then be escalated to level four. Inspired by the timeline of self-driving cars and ever-growing technologies, the work has taken up Carla simulator as a base for the problem statement formulation. It is open-source and provides various functionalities and closely relates to the real- world self-driving car scenarios. This repo branches out to deal with visual perception, state estimation and planning the motion of an autonomous vehicle. It also includes: generation of data and annotating them if necessary, analyzing the acquired data and making decisions accordingly, thus controlling and guiding the vehicle to manoeuvre wisely and enforcing quick learning and versatility by reducing the dependency on self-identification sensors.
</p>

## Setup.
<p style="text-align: justify;">
&nbsp;&nbsp;&nbsp;&nbsp; Refer to the <a href="https://carla.readthedocs.io/en/stable/getting_started/">Carla simulator document</a> to setup the simulator on your system. Once the the setup is complete install the necessary libraries and place the clone repository in '/CARLA_version/PythonAPI/' directory.
</p>

## Methodology.

## Directories and files.
<p style="text-align: justify;">
&nbsp;&nbsp;&nbsp;&nbsp; The repository has two main wings: the data generator and the simulation scripts. The control flows from the 'main.py' in the repository's root directory to either of these branched depending upon the command line arguments passed during execution. Files and folders are organized as follows:
<ul>
    <li>
        __DATA__
    </li>
    <li>
        __LOGS__
    </li>
    <li>
        __MODELS__
    </li>
    <li>
        DATA_SCRIPTS
    </li>
    <li>
        SIMULATION_SCRIPTS
    </li>
    <li>
        config.py
    </li>
    <li>
        main.py
    </li>
    <li>
        requirements.txt
    </li>
</ul>
</p>

## How to run?
## Results.
## References.
* [Capture data while manually controlling the agent.](https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py)