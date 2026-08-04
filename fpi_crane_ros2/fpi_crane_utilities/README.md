# fpi_crane_utilities

ROS 2 utility package for recording crane data, monitoring trajectory tracking,
and plotting tracking error CSV files.

## Package contents

- `fpi_crane_utilities/trajectory_tracking_monitor.py`: helper node that
  republishes the active trajectory point and logs joint tracking error
- `launch/record_ros_bag.launch.py`: launch file for recording common crane,
  camera, and lidar topics to a rosbag
- `cli/plot_trajectory_tracking_csv.py`: plotting helper for CSV files produced
  by `trajectory_tracking_monitor`
time dependencies

## Build

From the workspace root:

```bash
colcon build --packages-select fpi_crane_utilities
source install/setup.bash
```

If you are building the utility package with the hardware package it imports:

```bash
colcon build --packages-select fpi_crane_msgs fpi_crane_hw fpi_crane_utilities
source install/setup.bash
```

## Monitor trajectory tracking error

The `trajectory_tracking_monitor` node republishes the active trajectory point
as `/joint_setpoint` and publishes measured joint error as
`/jooint_tracking_error`. Both topics use `sensor_msgs/msg/JointState`; error
positions are computed as `/joint_states - /joint_setpoint`.

Run the monitor:

```bash
ros2 run fpi_crane_utilities trajectory_tracking_monitor
```

The monitor stores the latest `/RobotTrajectoryInfo` message and uses
`/plc_status.plc_crane_move_point_id` as the active 1-based trajectory point.
For each new `/RobotTrajectoryInfo` message, it opens a long-form CSV in
`/workspace/bags/errors` named with the current date and time plus
`_tracking_errors.csv`. Rows are written only while the matching trajectory is
executing.

Change the output directory:

```bash
ros2 run fpi_crane_utilities trajectory_tracking_monitor --ros-args \
  -p csv_directory:=/tmp/tracking_errors
```

Disable CSV logging:

```bash
ros2 run fpi_crane_utilities trajectory_tracking_monitor --ros-args \
  -p csv_directory:=
```

## Monitor topics

Subscribes:

- `/RobotTrajectoryInfo` (`fpi_crane_msgs/msg/RobotTrajectoryInfo`)
- `/plc_status` (`fpi_crane_msgs/msg/PlcStatus`)
- `/joint_states` (`sensor_msgs/msg/JointState`)
- `/rrc` (`sensor_msgs/msg/Joy`)

Publishes:

- `/joint_setpoint` (`sensor_msgs/msg/JointState`)
- `/jooint_tracking_error` (`sensor_msgs/msg/JointState`)

## CSV format

The tracking CSV contains one row per joint per sample. Columns are:

```text
time_sec, elapsed_sec, sequence_id, point_id, joint_name, target_position,
actual_position, position_error, actual_rrc, target_velocity, actual_velocity,
velocity_error, target_acceleration, actual_effort
```

The `actual_rrc` column is populated from `/rrc` for the PLC trajectory joints:

- `slew_joint`
- `boom_joint`
- `stick_joint`
- `telescope_joint`
- `grapplecarrier_joint`

## Plot tracking CSV files

Plot one joint from a saved CSV:

```bash
python3 fpi_crane_utilities/cli/plot_trajectory_tracking_csv.py \
  /workspace/bags/errors/20260617_153000_000000_tracking_errors.csv boom_joint
```

Save the plot instead of opening a window:

```bash
python3 fpi_crane_utilities/cli/plot_trajectory_tracking_csv.py \
  /workspace/bags/errors/20260617_153000_000000_tracking_errors.csv boom_joint \
  --output /tmp/boom_tracking.png
```

Plot all tracked joints:

```bash
python3 fpi_crane_utilities/cli/plot_trajectory_tracking_csv.py \
  /workspace/bags/errors/20260617_153000_000000_tracking_errors.csv all_joints
```

## Record crane rosbag data

Record the default crane, camera, and lidar topics:

```bash
ros2 launch fpi_crane_utilities record_ros_bag.launch.py
```

Set the location and test labels used in the bag name:

```bash
ros2 launch fpi_crane_utilities record_ros_bag.launch.py \
  location:=cranelab test:=calibration
```

The launch file records to a bag named:

```text
rosbag2_<location>_<test>_<YYYY_MM_DD-HH_MM_SS>
```

## Recorded topics

General crane topics:

- `/tf`
- `/tf_static`
- `/joint_states`
- `/plc_status`
- `/RobotTrajectoryInfo`

Stick camera topics:

- `/zed_1/zed_node/rgb/color/rect/camera_info`
- `/zed_1/zed_node/rgb/color/rect/image`
- `/zed_1/zed_node/point_cloud/cloud_registered`

Mast camera topics:

- `/zed_0/zed_node/rgb/color/rect/camera_info`
- `/zed_0/zed_node/rgb/color/rect/image`
- `/zed_0/zed_node/point_cloud/cloud_registered`

Mast lidar topics:

- `/lidar_0/points`
