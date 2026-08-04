# fpi_crane_rl

Learned point-cloud grasping policy for the crane lab. A PointNet policy
selects a grasp target from the ZED depth image, a 7-phase pick-and-place
FSM sequences the motion, and per-phase joint deltas are handed to the
`relative_joint_move` service for trajectory generation and PLC playback.

## Package layout

```
fpi_crane_rl/
  crane_policy_node.py     depth, PointNet policy, FSM, IK, service calls
  fsm.py                   pick-and-place state machine
  pointcloud_pipeline.py   depth + intrinsics to a 1024-point cloud (base frame)
  ik.py                    IKPy chain for slew/boom/stick/telescope
  policy_loader.py         loads bcrl, rl, sac, or heuristic policies
  pointnet_actor_critic.py PointNet actor-critic used by bcrl and rl
checkpoints/
  model_350.pt             trained BC+RL policy
urdf/
  fpiforwarder-upperpassive.urdf   URDF used by the IK chain (see TODO below)
```

## Build

Place the package in a colcon workspace alongside `fpi_crane_msgs`,
`fpi_crane_description`, `fpi_crane_hw`, then:

```bash
colcon build --packages-select fpi_crane_rl
source install/setup.bash
```

## Run

The PLC node (`fpi_crane_hw plc_node`) and the relative joint mover
(`fpi_crane_hw relative_joint_mover`) must both be running. Together they
own `/joint_command`, publish `/joint_states` and `/plc_status`, and
provide the `relative_joint_move` and `request_grapple_move` services.

```bash
ros2 run fpi_crane_rl crane_policy_node --ros-args \
  -p policy_type:=bcrl \
  -p checkpoint_path:=<path>/model_350.pt
```

## How it works

`crane_policy_node` runs the policy once per cycle to pick a grasp
target, then sequences the pick-and-place one phase at a time:

- For a motion phase: IK is solved for the Cartesian end-effector goal
  via `ik.CraneIK`, joint deltas are computed from the current joint
  state, and a `RelativeJointMove` service call is issued to
  `relative_joint_mover` (in `fpi_crane_hw`). That node generates a
  smooth 7th order trajectory and publishes
  `fpi_crane_msgs/RobotTrajectoryInfo` to the PLC. The policy waits for
  `/plc_status` to return to `HOLD` and for the joints to converge
  inside tolerance before advancing.
- For grasp and release: the `request_grapple_move` service is called
  directly and the policy waits for it to complete.

The same node drives both the simulator and the real crane, because the
PLC node presents the same interface in either mode.

## TODO

The URDF under `urdf/` is a truncated copy that stops at the
`upperpassive` link and uses structural joint names (for example
`basemast_to_mast` instead of `slew_joint`). It is loaded only by
`ik.CraneIK` for the IKPy chain, where joint names do not matter and the
chain ending matches the geometric IK target. For consistency with the
rest of the stack the IK should eventually load the URDF generated from
`fpi_crane_description/urdf/fpi_crane.urdf.xacro`. That requires
retargeting `base_elements`, verifying the chain link order, and
re-validating `EE_OFFSET_Z` against the new chain ending.
