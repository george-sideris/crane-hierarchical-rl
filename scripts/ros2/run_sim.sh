#!/bin/bash
# Launch sim_ros2_env.py with all required env vars
export LD_LIBRARY_PATH=/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/isaac-sim/exts/isaacsim.ros2.bridge/humble/rclpy:/workspace/crane_testbed/source/crane_testbed:$PYTHONPATH
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
cd /workspace/isaaclab
./isaaclab.sh -p /workspace/crane_testbed/scripts/ros2/sim_ros2_env.py --enable_cameras "$@"
