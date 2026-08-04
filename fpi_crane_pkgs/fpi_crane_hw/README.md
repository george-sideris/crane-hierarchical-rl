# plc_bridge_ros2

ROS 2 Python conversion of the ROS 1 PLC UDP bridge.

## What changed from ROS 1

- `rospy` was replaced with `rclpy`.
- Blocking `while not rospy.is_shutdown()` loops were replaced with ROS 2 timers.
- ROS 1 parameter access was replaced with ROS 2 declared parameters.
- ROS 1 service callback style was replaced with ROS 2 `(request, response)` callbacks.
- The old `moveit_commander` dependency was removed. Joint names and grapple limits are now parameters, which is usually simpler for a bridge node.
- The UDP socket helper was kept ROS-independent and updated with send-side ACK timeouts.

## Assumptions

This package assumes your custom ROS 2 interfaces are available in `robot_msgs` with these names:

- `robot_msgs/msg/PlcStatus.msg`
- `robot_msgs/msg/RobotTrajectoryInfo.msg`
- `robot_msgs/msg/PlcGripper.msg`
- `robot_msgs/msg/PlcStop.msg`
- `robot_msgs/srv/RequestGrappleMove.srv`

A small compatibility import shim is included for older lowercase generated names, but proper ROS 2 interface names should use PascalCase type names and snake_case field names.

## Build

Place this package in your ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cp -r plc_bridge_ros2 ~/ros2_ws/src/
cd ~/ros2_ws
colcon build --packages-select plc_bridge_ros2
source install/setup.bash
```

## Run in hardware mode

```bash
ros2 launch plc_bridge_ros2 plc_node.launch.py \
  use_simulation:=false \
  udp_recv_ip_address:=172.20.230.162 \
  udp_recv_port:=30305 \
  udp_send_ip_address:=172.20.230.120 \
  udp_send_port:=30310
```

## Run in simulation mode

```bash
ros2 launch plc_bridge_ros2 plc_node.launch.py use_simulation:=true
```

## Run the PLC emulator

```bash
ros2 run plc_bridge_ros2 plc_emulator --ros-args \
  -p src_ip:=172.20.230.120 \
  -p src_port:=30310 \
  -p remote_ip:=172.20.230.162 \
  -p remote_port:=30305
```

## Topics and service

Publishes:

- `/joint_states` (`sensor_msgs/msg/JointState`)
- `/joint_states_passive` (`sensor_msgs/msg/JointState`)
- `/plc_status` (`robot_msgs/msg/PlcStatus`)
- `/rrc` (`sensor_msgs/msg/Joy`)
- `/joint_command` in simulation mode only (`sensor_msgs/msg/JointState`)

Subscribes:

- `/RobotTrajectoryInfo` (`robot_msgs/msg/RobotTrajectoryInfo`)
- `/plc_stop` (`robot_msgs/msg/PlcStop`)
- `/joint_states_sim` in simulation mode only (`sensor_msgs/msg/JointState`)

Service:

- `request_grapple_move` (`robot_msgs/srv/RequestGrappleMove`)

## Parameters

Main parameters:

- `use_simulation` default `false`
- `udp_recv_ip_address` default `172.20.230.162`
- `udp_recv_port` default `30305`
- `udp_send_ip_address` default `172.20.230.120`
- `udp_send_port` default `30310`
- `joint_names`
- `passive_joint_names`
- `min_grapple_angle`
- `max_grapple_angle`
- `grapple_cmd_time_ms_sim`
- `grapple_velocity_sim`
- `telescope_scale_factor`
