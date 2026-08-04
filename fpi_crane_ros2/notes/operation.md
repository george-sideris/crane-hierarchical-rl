# Operation of Crane Stack

detailed notes on the operation of the crane stack developed for running the crane in Isaac Sim or on the real hardware with ros2 communications

"hw" stands for "hardware"

## fpi_crane_bringup

The lauch file `crane_bringup.launch.py` launches RViz and the PLC node. The PLC node has specific launch instructions defined in `plc_node.launch.py` such as ports and ip addresses. 

## fpi_crane_description

Contains the crane URDF/Xacro, visual meshes, sensor frames, and an RViz configuration for inspecting the robot state.

`description.launch.py` launches the robot_state_publisher and RViz with the robot description. Robot_state_publisher does not appear to be an executable script in the repo. How does this work? Can a launch.py file simply create a publisher with `launch_ros.actions.Node`?

## fpi_crane_hw

ROS 2 interface to the crane PLC and Isaac Sim simulator.

"udp" stands for "User Datagram Protocol". The `udp_handler.py` script defines communication parameters and methods for the messages being sent from the PLC (ie. max data size, packing data, unpacking jointstate messages from the various sensors, etc.)

PLC node can be luached seperately from the "bringup" laucher with `plc_node.launch.py`

## fpi_crane_msgs

contains msg and srv definitions. Specifically:

msg:
- PlcGripper.msg
- PlcStatus.msg
- PlcStop.msg
- RobotTrajectoryInfo.msg

srv:
- RelativeJointMove.srv
- RequestGrappleMove.srv

## fpi_crane_rl

Learned point-could based grasping policy for the crane lab.

`crane_policy_node.py` uses RelativeJointMove.srv, and RequestGrappleMove.srv as services now for executing different fsm state transitions. The legacy implementation used SetTarget.srv and CheckTargetReached.srv service structure. 

## Filing structure

    ~/FPI_liebherr_automation/
    ├── Omniverse-Simulation   # Assets and scene for simulation       
    └── ros2_ws/   # Build in here
        └── src/
            ├── fpi_crane_perception_kit   # Azure repo
            ├── fpi_crane_ros2    
            ├── etc.

packages are moved to the `src` directory within the new workspace to be built

# Running in Isaac Sim

Instructions for running code stack with the crane screen in Isaac Sim. 

## Isaac Sim

Launch Isaac Sim using the workstation install (ensure ros2 install is not sourced in terminal running Isaac Sim):

```bash
export isaac_sim_package_path=$HOME/isaacsim
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$isaac_sim_package_path/exts/isaacsim.ros2.bridge/humble/lib
~/isaacsim/isaac-sim.sh
```
Open the `crane_lab_gaze.usd` scene (or the most recent version of the scene) and play.

Topics published by the Isaac Sim scene:

    /basemast/camera_info
    /basemast/depth
    /basemast/imu/data
    /basemast/rgb
    /clock
    /joint_command
    /joint_states_sim
    /parameter_events
    /rosout
    /tf
    /zedx/camera_info
    /zedx/depth
    /zedx/rgb

The basemast camera is equivalent to `zed_0` in on the real crane.

## Hardware and RL packages

Ros2 packages that facilitate the operation of the crane and process outputs from Isaac Sim

Source ros2 and package install:

```bash
source /opt/ros/humble/setup.bash
source ~/FPI_liebherr_automation/ros2_ws/install/setup.bash
```

### Bringup (PLC, relative joint move, robot description, etc.)

Launch the crane bringup:
```bash
ros2 launch fpi_crane_bringup crane_bringup.launch.py \
  sim:=true \
  use_sim_time:=true \
  viz:=true \
  joy:=false
```
This runs the plc node, starts publishing the crane description, and opens rviz2. Change to the correct rviz2 config to stream the sim.

The PLC node in sim mode (`sim:=true`) emulates how the real PLC processes encoder data packets and outputs them to the rest of the code stack. 

- plc_node subscribes to `/joint_states_sim`
- plc_node publishes `/joint_command`
- plc_node republishes `/joint_states`
- relative_joint_mover still provides `/relative_joint_move`

> In theory the middle joint states process in the PLC node is not necessary since the Isaac Sim Action Graph is publishing this information directly to ros2 topics. 

### Stability Action Server

Launch the stability action server with the simulated IMU topic:

```bash
ros2 run fpi_crane_rl_metrics measure_stability_action_node --ros-args \
  -p use_sim_time:=true \
  -p base_frame:=base_link \
  -p grapple_frame:=basegrapple \
  -p imu_topic:=/basemast/imu/data
```

### Policy

Activate a virtual environment with all numpy dependencies for the policy installed (pytorch, ikpy, etc.)

> With a workstation install of ros2 this is necessary since the ros2 python3 version does not support some packages used to run the policy. This makes it so you cannot run the policy with `ros2 run`. `python3 <file>` must be used instead.

```bash
source ~/crane_venv/bin/activate
```

Change directories to the crane rl package and run policy node:

```bash
cd ~/FPI_liebherr_automation/ros2_ws/src/fpi_crane_ros2/fpi_crane_rl/fpi_crane_rl

python3 crane_policy_node.py --ros-args \
  -p use_sim_time:=true \
  -p policy_type:=bcrl \
  -p checkpoint_path:=/home/fpiadmin/FPI_liebherr_automation/ros2_ws/src/fpi_crane_ros2/fpi_crane_rl/checkpoints/model_350.pt \
  -p measure_stability_enabled:=true \
  -p traj_velocity:=0.5 \
  -p traj_min_duration:=0.1 \
  -p yaw_velocity:=2.0 \
  -p yaw_min_duration:=0.1 \
  -p telescope_speed:=0.5 \
  -r /zedx/points:=/basemast/points \
  -r /zedx/camera_info:=/basemast/camera_info
```

### Keyboard Joystick Control

Run the keyboard joystick node to control the sim crane using a keyboard:
```bash
ros2 run fpi_crane_hw keyboard_joy_node
```

> Change default "teleop_scale_vel" ros parameter in `plc_node.py` to change gain on teleop input

TODO: Make per joint gains in `plc_node.py` 

# Running on Real Hardware

Instructions for operating real hardware. Have the designated operator (Heshan) start the crane and ramp up engine. 

### IP addresses

- Main PC - 172.20.230.162
- PLC - 172.20.230.120
- Jetson - 172.20.230.118
- LiDAR - 172.20.230.121

- Secondary PC - 172.20.230.132

## Main PC

The main PC controls plc -> ros communications nodes, motion planning nodes, policy sequencing nodes, visualization, and data logging.

### terminal 1: Jetson

This terminal connects to the jetson to start the camera  data streams. The jetson only handles camera data in this configuration of the hardware. This was done to fix ros2 communications problems to the PC ending policy commands and logging data. 

> Ensure that the jetson's time is matched with the main PC (.162 ip address). If not, the data streams will not be available in rviz. 

Send time from main PC to jetson

```bash
ssh -t user@172.20.230.118 "sudo date -s '$(date +"%Y-%m-%d %H:%M:%S")'"
```

SSH into jetson and start streaming camera data through ros

```bash
# connect to jetson through ssh
ssh user@172.20.230.118 #password: admin

# reboot jetson
sudo reboot run

# restart cameras
sudo systemctl restart zed_x_daemon

# run camera
python3 streaming_senders.py

```
### terminal 2: plc node (bringup launch)

In a second terminal run the bringup launch file. This launch file starts the PLC node and RViz with a custom configuration for the data streams of the crane.

```bash
source /opt/ros/humble/setup.bash
source ~/FPI_liebherr_automation/ros2_ws/install/setup.bash

ros2 launch fpi_crane_bringup crane bringup.launch.py \
  sim:=false \
  viz:=true \ 
  joy:=false
```
This terminal will continue running the PLC node which faciliates publishing joint states, and other data packets sent from the PLC. Errors from the PLC node will be printed in this terminal, read to debug issues. 

### terminal 3: perception kit

In another terminal, launch the perception kit. Launching the perception packages will run nodes that publish the camera data to ros.

```bash
source /opt/ros/humble/setup.bash
source ~/FPI_liebherr_automation/ros2_ws/install/setup.bash

ros2 launch fpi_crane_perception_kit perception.launch.py \
  launch_zed_0:=true \
  launch_zed_1:=false \
  launch_lidar_0:=false \
  zed_streaming:=true
```
Use `zed_streaming:=false` only if the ZED is local USB rather than network stream.

Read prints in this terminal to ensure perception package is functioning porperly. 

> launching the perception kit requires installation of the ZED SDK. See README.md in root of perception kit for more information. 

### terminal 4: stability action server

run the stability action server

```bash
ros2 run fpi_crane_rl_metrics measure_stability_action_node --ros-args \
  -p base_frame:=base_link \
  -p grapple_frame:=grapplecarrier \
  -p imu_topic:=/zed_0/zed_node/imu/data
```

### terminal 5: policy

Activate a virtual environment with all numpy dependencies for the policy installed (pytorch, ikpy, etc.)

> With a workstation install of ros2 this is necessary since the ros2 python3 version does not support some packages used to run the policy. This makes it so you cannot run the policy with `ros2 run`. `python3 <file>` must be used instead.

```bash
source ~/crane_venv/bin/activate
```

Change directories to the crane rl package from home and run the policy node with all necessary `ros-args` defined

```bash
cd ~/FPI_liebherr_automation/ros2_ws/src/fpi_crane_ros2/fpi_crane_rl/fpi_crane_rl

python3 crane_policy_node.py --ros-args \
  -p policy_type:=bcrl \
  -p checkpoint_path:=/home/fpiadmin/FPI_liebherr_automation/ros2_ws/src/fpi_crane_ros2/fpi_crane_rl/checkpoints/model_350.pt \
  -p measure_stability_enabled:=true \
  -p measure_stability_action_name:=measure_stability
```

## Container framework

Alternatively, the `bringup`and `perception_kit` can be run in a ros dedicated container.

The `fpi_ros2_devcontainer` in configured for running these nodes. 

> Ensure that the ZED SDK version in the docker container image matches your system install or you will not be able to run the perception kit within the container. 

> This container is a work-in-progress. It could still use debugging of the compatability with Cyclone DDS which is optimized for communication of ZED camera information.  

```bash
docker compose -f docker-desktop-compose.yml up -d

docker exec -it fpi_ros2_container bash

apt update
apt install ros-humble-rmw-cyclonedds-cpp
echo $RMW_IMPLEMENTATION # rmw_cyclonedds_cpp

## Relative Joint Move

Joint limits of crane hardware:
```python
JOINT_LIMITS = [
        (-1.74533,  1.74533),   # slew
        (-0.383972, 1.309),     # boom
        (-3.08574,  0.035),     # stick
        ( 0.13,     1.8),       # telescope
    ]
```

Joint names in PLC node:

```python
DEFAULT_JOINT_NAMES = [
    "slew_joint",
    "boom_joint",
    "stick_joint",
    "telescope_joint",
    "hanger_joint",
    "bearingfork_joint",
    "grapplecarrier_joint",
    "grappletong1_joint",
    "grappletong2_joint",
]
```

/joint_states topic at slew max before sensor is damaged:

    position:
    - 1.8186315444444443
    - 0.7976147944444444
    - -1.3840449277777775
    - 0.7525
    - 0.6300633277777776
    - -0.07853974999999999
    - -0.019198605555555553
    - 0.0
    - 0.0

Note that the joints can move further than the URDF limits. The URDF limits are defined for SIM. They also represent a useful safety threshold.

    Slew_max_urdf ~ 100deg

### /relative_joint_mover service call

```bash
ros2 service call /relative_joint_move fpi_crane_msgs/srv/RelativeJointMove "{
  sequence_id: 0,
  joint_names: ['slew_joint', 'boom_joint', 'stick_joint', 'telescope_joint', 'grapplecarrier_joint'],
  delta_positions: [1.5, 0.5, 0.5, 0, 0],
  velocity: 5.0,
  min_duration: 10.0
}"
```

# Notes

## Domain IDs


```


TODO:


