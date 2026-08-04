#!/usr/bin/env bash
# Record a crane trial rosbag to the T9 SSD via a throwaway ros2 container,
# then restamp the ZED topics (kit clock offset) automatically.
# Usage: ./record_trial.sh [name_prefix]   (default: crane_run)
# Stop recording with Ctrl-C; the restamp runs right after.
# NOTE: topic selection must be ONE -e regex; mixing explicit topic names
# with -e makes humble's recorder subscribe to nothing.
set -euo pipefail

PREFIX="${1:-crane_run}"
[ -d /media/george/T9 ] || { echo "T9 SSD not mounted" >&2; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
BAG="/media/george/T9/${PREFIX}_${STAMP}"

docker run --rm -it --network host \
  -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e ACCEPT_EULA=Y --entrypoint bash \
  -v /media/george/T9:/mnt/ssd \
  -v /home/george/IsaacLab/crane_testbed:/workspace/crane_testbed \
  isaac-lab-ros2 \
  -c "source /opt/ros/humble/setup.bash && \
      source /workspace/crane_testbed/fpi_sandbox_ws/install/setup.bash && \
      ros2 bag record -o /mnt/ssd/${PREFIX}_${STAMP} \
      --max-cache-size 2147483648 \
      -e '(/crane/.*|/pile_analyzer_server/debug/markers/.*|/pile_analyzer_server/dev/attack_pose|/joint_states|/tf|/plc_status|/plc_stop|/RobotTrajectoryInfo|/rrc|/zed_0/zed_node/(point_cloud/cloud_registered|imu/data|rgb/color/rect/image/compressed|rgb/color/rect/camera_info))'" \
  || true

if [ -f "$BAG/metadata.yaml" ]; then
  echo "recording done, restamping ZED topics ..."
  python3 "$(dirname "$0")/clock_sync/restamp_zed_bag.py" "$BAG" --latency 0.06 \
    && echo "use ${BAG}_restamped for anything stamp-aligned" \
    || echo "restamp failed; run clock_sync/restamp_zed_bag.py on $BAG manually"
else
  echo "no bag written at $BAG (recorder closed without metadata)" >&2
fi
