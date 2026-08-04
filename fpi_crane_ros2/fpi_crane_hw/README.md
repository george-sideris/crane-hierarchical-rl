# fpi_crane_hw

ROS 2 interface to the crane PLC and Isaac Sim simulator.

## Package contents

- `fpi_crane_hw/plc_node.py`: main ROS 2 PLC bridge node
- `fpi_crane_hw/udp_handler.py`: UDP protocol send/receive helper for PLC communication
- `fpi_crane_hw/robot_globals.py`: shared PLC, gripper, joint, and RRC enum values
- `fpi_crane_hw/plc_emulator.py`: simple PLC UDP emulator for local testing
- `fpi_crane_hw/relative_joint_mover.py`: helper node that turns relative joint service calls into trajectory messages
- `launch/plc_node.launch.py`: launch file for hardware and simulation modes
- `test/send_udp_data_from_file.py`: utility for replaying captured UDP data
- `test/20241022_datastream.txt`: sample captured PLC data stream

## Build

From the workspace root:

```bash
colcon build --packages-select fpi_crane_hw
source install/setup.bash
```

## Run in hardware mode

```bash
ros2 launch fpi_crane_hw plc_node.launch.py
```

## Run in simulation mode

```bash
ros2 launch fpi_crane_hw plc_node.launch.py use_simulation:=true
```

## Topics and service

Publishes:

- `/joint_states` (`sensor_msgs/msg/JointState`)
- `/plc_status` (`fpi_crane_msgs/msg/PlcStatus`)
- `/rrc` (`sensor_msgs/msg/Joy`)
- `/joint_command` in simulation mode only (`sensor_msgs/msg/JointState`)

Subscribes:

- `/RobotTrajectoryInfo` (`fpi_crane_msgs/msg/RobotTrajectoryInfo`)
- `/plc_stop` (`fpi_crane_msgs/msg/PlcStop`)
- `/joint_states_sim` in simulation mode only (`sensor_msgs/msg/JointState`)

Service:

- `request_grapple_move` (`fpi_crane_msgs/srv/RequestGrappleMove`)

## PLC commands from ROS 2

The PLC node accepts commands through ROS 2 topics and services. Start the node in hardware or simulation mode first, then use the interfaces below.

### Send a crane trajectory using the relative joint mover

The `relative_joint_mover` helper can generate and publish `/RobotTrajectoryInfo` messages from relative joint movements.

Run the helper (if not running the launch file):

```bash
ros2 run fpi_crane_hw relative_joint_mover
```

Then call `/relative_joint_move`:

```bash
ros2 service call /relative_joint_move fpi_crane_msgs/srv/RelativeJointMove "{
  sequence_id: 0,
  joint_names: ['boom_joint'],
  delta_positions: [1.5],
  velocity: 5.0,
  min_duration: 10.0,
  trajectory_type: 'smooth'
}"
```

Use `trajectory_type: 'step'` to publish a constant trajectory at the target value,
or `trajectory_type: 'ramp'` to publish a linear ramp from the current position to the target.

The node accepts a trajectory only when the PLC crane state is `HOLD`.

### Stop trajectory execution

Publish a `fpi_crane_msgs/msg/PlcStop` message to `/plc_stop`.

```bash
ros2 topic pub --once /plc_stop fpi_crane_msgs/msg/PlcStop "{stop_traj_execution: true}"
```

In hardware mode this sends the PLC stop command over UDP. In simulation mode it stops the simulated trajectory or gripper execution.

### Open or close the gripper

Call the `request_grapple_move` service with a `fpi_crane_msgs/srv/RequestGrappleMove` request.

Open:

```bash
ros2 service call /request_grapple_move fpi_crane_msgs/srv/RequestGrappleMove "{
  grapple_move: {
    sequence_id: 10,
    is_executed: false,
    gripper_move_to: 1,
    delta_time: 4000,
    delta_angle: 0
  }
}"
```

Close:

```bash
ros2 service call /request_grapple_move fpi_crane_msgs/srv/RequestGrappleMove "{
  grapple_move: {
    sequence_id: 11,
    is_executed: false,
    gripper_move_to: 2,
    delta_time: 4000,
    delta_angle: 0
  }
}"
```

`gripper_move_to` values:

- `0`: none
- `1`: open
- `2`: close

### Watch PLC status

```bash
ros2 topic echo /plc_status
```

Useful status fields:

- `plc_crane_state`: `0` INIT, `1` HOLD, `2` EXECUTE_SEQUENCE, `3` END_SEQUENCE
- `plc_crane_sequence_id`
- `plc_crane_move_point_id`
- `plc_trajectory_count_in_queue`
- `plc_gripper_state`
- `plc_gripper_move_completeness`
