# fpi_crane_bringup

ROS 2 bringup package for launching the FPI crane stack. It coordinates the robot description, PLC bridge, and optional joystick input from one launch file.

## Package contents

- `launch/crane_bringup.launch.py`: top-level launch file for the crane stack
- `fpi_crane_bringup/__init__.py`: Python package marker
- `setup.py`: package installation metadata

## Build

From the workspace root:

```bash
colcon build --packages-select fpi_crane_bringup
source install/setup.bash
```

If you are building the normal crane stack:

```bash
colcon build --packages-select fpi_crane_description fpi_crane_hw fpi_crane_msgs fpi_crane_bringup
source install/setup.bash
```

## Launch the crane stack

Launch the default hardware stack:

```bash
ros2 launch fpi_crane_bringup crane_bringup.launch.py
```

Launch in simulation mode:

```bash
ros2 launch fpi_crane_bringup crane_bringup.launch.py sim:=true
```

Launch without RViz:

```bash
ros2 launch fpi_crane_bringup crane_bringup.launch.py viz:=false
```

Launch with joystick input:

```bash
ros2 launch fpi_crane_bringup crane_bringup.launch.py joy:=true
```

## What gets launched

The bringup launch file includes:

- `fpi_crane_description/description.launch.py`
- `fpi_crane_hw/plc_node.launch.py`
- `joy/joy_node`, only when `joy:=true`

## Perception hardware

For the real crane, the perception sensors (zedx, lidar) need to be launched separately with the `fpi_crane_perception_stack`