#!/usr/bin/env bash
# Floor-constrained camera calibration from a single motion rosbag.
# For each camera: crop grapple frames -> grapple model->cloud fit + floor-levelness.
# Writes out/calib_<cam>_floor_{mount,params}.npy (review, then promote manually).
#
# Usage: ./run_floor_calib.sh <bag_dir> [cam ...]
#   ./run_floor_calib.sh ../rosbag2_cranelab_test_2026_06_26-16_48_33
#   ./run_floor_calib.sh ../some_bag zed_0          # single camera
set -euo pipefail
cd "$(dirname "$0")"

BAG="${1:?usage: run_floor_calib.sh <bag_dir> [cam ...]}"
shift || true
CAMS=("$@"); [ ${#CAMS[@]} -eq 0 ] && CAMS=(zed_0 zed_1)
[ -d "$BAG" ] || { echo "bag not found: $BAG" >&2; exit 1; }

for CAM in "${CAMS[@]}"; do
  echo "=================  $CAM  ================="
  echo "[1/3] exporting grapple frames from bag ..."
  python3 export_grapple.py --bag "$BAG" --cam "$CAM" --step 8 --out "out/grapple_${CAM}.npz"
  echo "[2/3] floor-constrained fit ..."
  python3 combined_calib.py --cam "$CAM" --floor-bag "$BAG" --npz "out/grapple_${CAM}.npz"
  echo "[3/3] reprojection validation (green=new calib, red=nominal) ..."
  python3 reproject_validate.py --cam "$CAM" --npz "out/grapple_${CAM}.npz" --bag "$BAG" \
      --params "out/calib_${CAM}_floor_params.npy" --out "out/reproj_${CAM}_floor.jpg" || \
      echo "  (reproj skipped: no rgb frames matched in bag)"
  # archive the reproj image under the same timestamp combined_calib gave the npy fit,
  # so reproj_<cam>_<stamp>_floor.jpg <-> calib_<cam>_<stamp>_floor_params.npy stay linked.
  STAMP=$(ls -t out/calib_${CAM}_*_floor_params.npy 2>/dev/null | head -1 \
          | sed -E "s|.*/calib_${CAM}_([0-9_]+)_floor_params\.npy|\1|")
  if [ -n "$STAMP" ] && [ -f "out/reproj_${CAM}_floor.jpg" ]; then
    cp "out/reproj_${CAM}_floor.jpg" "out/reproj_${CAM}_${STAMP}_floor.jpg"
    echo "  archived: out/reproj_${CAM}_${STAMP}_floor.jpg"
  fi
  echo
done

echo "done. results in out/calib_<cam>_floor_mount.npy (NOT yet promoted)."
echo "validation images: out/reproj_<cam>_floor.jpg (latest) + out/reproj_<cam>_<stamp>_floor.jpg (archived)."
echo "to promote a camera:  cp out/calib_<cam>_floor_mount.npy out/calib_<cam>_mount.npy"
echo "                      cp out/calib_<cam>_floor_params.npy out/calib_<cam>_params.npy"
echo "then re-bake the URDF joint for that camera."
