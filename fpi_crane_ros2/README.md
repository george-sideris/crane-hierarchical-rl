# fpi_crane_ros2

ROS 2 workspace packages for the FPI crane.

## Packages

- `fpi_crane_description`: URDF/Xacro, meshes, sensor frames, and RViz config for the crane model.
- `fpi_crane_msgs`: Custom ROS 2 messages and services used by the crane stack.
- `fpi_crane_hw`: PLC bridge, simulation support, PLC emulator, and relative joint movement helper.
- `fpi_crane_bringup`: Top-level launch package for starting the crane description, PLC bridge, and optional joystick input.

## Build

From the workspace root:

```bash
colcon build --symlink-install --packages-select fpi_crane_description fpi_crane_msgs fpi_crane_hw fpi_crane_bringup
source install/setup.bash
```

## Launch

Launch the hardware stack:

```bash
ros2 launch fpi_crane_bringup crane_bringup.launch.py
```

Launch in simulation mode:

```bash
ros2 launch fpi_crane_bringup crane_bringup.launch.py sim:=true
```
