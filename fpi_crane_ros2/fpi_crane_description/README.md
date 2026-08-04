# fpi_crane_description

ROS 2 description package for the FPI crane model. It contains the crane URDF/Xacro, visual meshes, sensor frames, and an RViz configuration for inspecting the robot state.

## Package contents

- `urdf/fpi_crane.urdf.xacro`: main crane model
- `urdf/sensors.xacro`: reusable sensor frame macros
- `meshes/`: STL visual meshes for the crane links
- `launch/description.launch.py`: launches `robot_state_publisher` and optionally RViz
- `rviz/config.rviz`: RViz configuration for the crane model

## Build

From the workspace root:

```bash
colcon build --packages-select fpi_crane_description
source install/setup.bash
```

## Launch the robot description

Launch the description with RViz:

```bash
ros2 launch fpi_crane_description description.launch.py
```

Launch without RViz:

```bash
ros2 launch fpi_crane_description description.launch.py viz:=false
```

Launch in simulation mode:

```bash
ros2 launch fpi_crane_description description.launch.py sim:=true
```

In hardware mode, `robot_state_publisher` listens to `/joint_states`. In simulation mode, the launch file remaps `joint_states` to `joint_states_sim`, so `robot_state_publisher` listens to `/joint_states_sim`.

## Launch arguments

- `viz`: default `true`; starts RViz with `rviz/config.rviz`
- `sim`: default `false`; enables the simulation joint state remap

## Published and consumed interfaces

The launch file starts `robot_state_publisher`.

Consumes:

- `/joint_states` (`sensor_msgs/msg/JointState`) in hardware mode
- `/joint_states_sim` (`sensor_msgs/msg/JointState`) in simulation mode

Publishes:

- `/robot_description` (`std_msgs/msg/String`-style parameter content exposed by `robot_state_publisher`)
- `/tf` (`tf2_msgs/msg/TFMessage`)
- `/tf_static` (`tf2_msgs/msg/TFMessage`)

## Inspect the generated URDF

View the TF tree while the launch file is running:

```bash
ros2 run tf2_tools view_frames
```

## Joint names

The primary crane joints are:

- `slew_joint`
- `boom_joint`
- `stick_joint`
- `telescope_joint`
- `hanger_joint`
- `bearingfork_joint`
- `grapplecarrier_joint`
- `grappletong1_joint`
- `grappletong2_joint`

These names match the default joint ordering used by `fpi_crane_hw`.

## Main links and frames

The model root is `base_link`. Main crane links include:

- `mast`
- `mainboom`
- `stick`
- `telescope`
- `upperpassive`
- `lowerpassive`
- `grapplecarrier`
- `grappletong1`
- `grappletong2`
- `trailer`

Additional fixed frames include:

- `camera_zed`
- `mast_sensor_bracket`
- `grappletong1_tip`
- `grappletong2_tip`

## Sensor frames

The model includes fixed sensor frames generated from `sensors.xacro`:

- `zed_1_camera_link`, mounted under `camera_zed`
- `lidar_0/os_sensor`, mounted under `mast_sensor_bracket`
- `zed_0_camera_link`, mounted under `lidar_0/os_sensor`

These are frame definitions only. Camera and lidar drivers are launched by the`fpi_crane_perception_kit` package.