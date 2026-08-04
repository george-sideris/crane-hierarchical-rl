# Source this in any system-Python terminal that runs jv_controller_node.py or
# crane_policy_node.py. Builds /workspace/crane_testbed/.crane_ws on first use
# (the build persists on the host mount, so subsequent container restarts skip
# the rebuild), then sets up ROS2 paths.
#
# Usage:  source /workspace/crane_testbed/scripts/ros2/env_setup.sh

WS=/workspace/crane_testbed/.crane_ws
SRC=/workspace/crane_testbed/scripts/ros2/sim_interface

if [ ! -f "$WS/install/setup.bash" ]; then
    echo "[env_setup] crane_ws not built yet — building sim_interface..."
    mkdir -p "$WS/src"
    ln -sfn "$SRC" "$WS/src/sim_interface"
    (
        unset PYTHONPATH LD_LIBRARY_PATH
        source /opt/ros/humble/setup.bash
        cd "$WS" && colcon build --packages-select sim_interface --symlink-install
    )
fi

unset PYTHONPATH LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
echo "[env_setup] ready: ROS2 humble + sim_interface (system python3.10)"
