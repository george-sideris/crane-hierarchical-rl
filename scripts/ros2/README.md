# Crane ROS2 Sim2Real Pipeline

ROS2 nodes that run a BC+RL grasping policy on the forestry crane testbed,
shared between the Isaac Lab simulator and the real hardware.

The policy node, FSM, and controller node are **hardware-agnostic** — they
consume and publish standard ROS2 topics. Swapping sim for real means
replacing `sim_ros2_env.py` with the equivalent driver-side topic providers
(ZED X driver, crane PLC bridge, URDF static TF). Nothing in the policy or
controller stack changes between the two.

## Components

```
scripts/ros2/
├── sim_ros2_env.py        # Isaac Lab sim + ROS2 bridge (sim-side only)
├── crane_policy_node.py   # depth → policy → FSM → EE targets
├── jv_controller_node.py  # EE target → joint commands (ikpy + PD)
├── fsm.py                 # 10-phase pick-and-place state machine
├── pointcloud_pipeline.py # depth + intrinsics + extrinsics → 1024-pt PCD
├── policy_loader.py       # loads bcrl / pure-rl / sac / heuristic policies
├── env_setup.sh           # builds .crane_ws and sources system Python ROS2
└── sim_interface/         # ROS2 package with custom service definitions
    └── srv/
        ├── SetTarget.srv          # xyz, yaw, grapple_opening
        └── CheckTargetReached.srv # tolerance → bool
```

## Topic + service flow

```
                     ┌──────────────────────────┐
                     │  sim_ros2_env.py  (sim)  │      (real hardware)
                     │  OR: ZED X + crane PLC   │
                     └──────────┬───────────────┘
                                │
            /zedx/depth, /zedx/camera_info, /tf_static
            /crane/joint_states, /crane/ee_pos_base
            /crane/grapple_yaw_base, /crane/action_bounds
                                │
                                ▼
                     ┌──────────────────────────┐
                     │  crane_policy_node.py    │
                     │  - point cloud pipeline  │
                     │  - policy forward        │
                     │  - 10-phase FSM (fsm.py) │
                     └──────────┬───────────────┘
                                │
                                │  /crane/set_target  (SetTarget service)
                                │     target_xyz, target_yaw, grapple_opening
                                ▼
                     ┌──────────────────────────┐
                     │  jv_controller_node.py   │
                     │  - ikpy IK on target_xyz │
                     │  - per-joint PD @ 10 Hz  │
                     └──────────┬───────────────┘
                                │
                                │  /crane/joint_command (sensor_msgs/JointState)
                                ▼
                     ┌──────────────────────────┐
                     │  sim_ros2_env.py         │
                     │  OR: crane PLC driver    │
                     └──────────────────────────┘
```

### What each node owns

**`sim_ros2_env.py`** (sim-side adapter; replaced by hardware drivers in deployment)
- Spawns Isaac Lab's `CranePointCloudDirectEnv` headless and a Replicator
  render product for the ZED X camera.
- Publishes everything the policy/controller stack reads: depth, camera info,
  joint states, basegrapple pose, static TF, and a `TRANSIENT_LOCAL`-latched
  `/crane/action_bounds` message containing the workspace AABB the policy was
  trained on.
- Subscribes to `/crane/joint_command` and applies it to Isaac Lab's articulation.
- Camera and base transforms are read from the USD scene **before** rclpy is
  initialized, because Isaac Sim's CUDA initialization conflicts with rclpy
  imports if the order is reversed.

**`crane_policy_node.py`** (the brain)
- Subscribes to depth + camera info + TF and runs `pointcloud_pipeline.process_depth()`
  to get a 1024-point PCD in the crane base frame.
- Calls the loaded policy (`policy_loader.load_policy(...)`) to get a grasp target
  `(x, y, z, yaw)`.
- Drives the `CraneFSM` (`fsm.py`) which sequences the 10 phases:
  `HOVER_UP → ALIGN_YAW → DESCEND → CLOSE → LIFT_HIGH → CARRY_HOME →
  ALIGN_HOME_YAW → LOWER_TO_DROP → OPEN → SETTLE → CLEAR`.
- Converts the FSM's basegrapple-frame yaw target to a yaw-joint target
  (`_grapple_yaw_to_joint`, ports `crane_rl_env_full.py:_convert_grapple_yaw_to_joint_position`).
- Sends each FSM step to the controller via the `SetTarget` service.

**`jv_controller_node.py`** (the body)
- Owns the joint-space PD loop at `LOOP_HZ = 10.0`.
- On `/crane/set_target`, runs ikpy IK on `target_xyz` to get joint targets
  (excluding yaw + gripper, which are commanded directly).
- Publishes `/crane/joint_command` with both position and velocity feed-forward.
- Also publishes `/crane/ee_state` (FK estimate, used only for legacy
  debugging) and `/crane/gripper_state`.

**`fsm.py`** (the conductor)
- Pure-Python state machine — no ROS2 dependencies, easy to unit test.
- Configurable via `FSMConfig` (`hover_clear`, `approach_above`, `ee_tolerance`,
  `drop_position`, etc.).
- Emits an `FSMCommand` each tick: `{position, yaw, gripper, phase, cycle_complete}`.

**`pointcloud_pipeline.py`** (the eyes)
- Mirrors Isaac Lab's PCD pipeline exactly: unproject → world → base-frame
  transform → depth filter `[1.0, 10.0] m` → farthest-point-sample to 1024 pts.
- Pure functions, easy to validate (`validate_pipeline.py`).

**`policy_loader.py`** (the model registry)
- `load_policy(policy_type, checkpoint, bounds_min, bounds_max, cossin, device)`
- Supports `bcrl`, `pure_rl`, `sac`, `heuristic`. The deployed BC+RL checkpoint
  uses the frozen-encoder variant — see `project_bcrl_deploy_checkpoint.md`
  for the rationale.

## Sim test (3 terminals, all inside the Docker container)

In **every** terminal, source the env first — it builds `.crane_ws` if needed
and unsets `PYTHONPATH`/`LD_LIBRARY_PATH` so the system rclpy doesn't collide
with Isaac Sim's bundled rclpy. (See `project_rclpy_split.md`.)

```bash
source /workspace/crane_testbed/scripts/ros2/env_setup.sh
```

**Terminal 1 — Isaac Lab sim + ROS2 bridge:**
```bash
PYTHONPATH=/workspace/crane_testbed/source/crane_testbed:$PYTHONPATH \
  ./isaaclab.sh -p /workspace/crane_testbed/scripts/ros2/sim_ros2_env.py
```
*(uses Isaac Sim's Python 3.11 + bundled rclpy; do NOT source env_setup.sh
in this terminal — it would unset the Isaac Sim PYTHONPATH it needs.)*

**Terminal 2 — joint-velocity controller:**
```bash
/usr/bin/python3 /workspace/crane_testbed/scripts/ros2/jv_controller_node.py
```

**Terminal 3 — policy node:**
```bash
/usr/bin/python3 /workspace/crane_testbed/scripts/ros2/crane_policy_node.py \
    --policy_type bcrl \
    --checkpoint /workspace/crane_testbed/checkpoints/model_350.pt
```

## Real deployment

The integrator must provide ROS2 publishers for every topic the policy node
and controller node read. Then run terminals 2 and 3 above against the real
topics — terminal 1 (the sim) is dropped.

| Topic / Service | Direction | Provider on real testbed |
|-----------------|-----------|--------------------------|
| `/zedx/depth` (Image, 32FC1, meters) | PUB | ZED X driver |
| `/zedx/camera_info` (CameraInfo) | PUB | ZED X driver |
| `/tf_static` | PUB | URDF + ZED extrinsic — frames `world`, `crane_base`, `zedx_camera` |
| `/crane/joint_states` (JointState) | PUB | Crane PLC bridge |
| `/crane/ee_pos_base` (PointStamped) | PUB | Crane PLC bridge — basegrapple position in `crane_base` |
| `/crane/grapple_yaw_base` (Float64) | PUB | Crane PLC bridge — basegrapple yaw in `crane_base`, radians |
| `/crane/action_bounds` (Float32MultiArray, 6 floats, latched) | PUB | Static publisher — `[min_x, min_y, min_z, max_x, max_y, max_z]` for the actual rack location |
| `/crane/joint_command` (JointState) | SUB | Crane PLC bridge — consumes position + velocity targets |
| `/crane/set_target` (SetTarget service) | provided by jv_controller_node | — |

### Notes for real deployment

- The frame ID used by the policy node defaults to `crane_base` (parameter
  `--base_frame`); the URDF / `static_transform_publisher` must match.
- The deployed checkpoint expects 1024-point PCDs in the `crane_base` frame,
  depth-filtered to `[1.0, 10.0] m`. `process_depth()` in
  `pointcloud_pipeline.py` handles the conversion.
- Action bounds **must** match what the policy was trained against. For the
  current checkpoint they are `min ≈ (-5.0, -0.75, -1.373)`,
  `max ≈ (-3.0, 4.59, -0.373)` (rack-centered AABB in base frame). If the
  rack location on the real testbed differs, the bounds need to shift
  accordingly — see `_compute_action_space_bounds` in
  `crane_rl_env_full.py` for the formula. The policy may need re-training
  if the bounds shift far from the trained range.
- The basegrapple has 180° yaw symmetry. The `_grapple_yaw_to_joint`
  conversion in the policy node picks the candidate joint position
  numerically closest to the current joint position to avoid 360° spins
  if the yaw joint has wrapped past ±π.

## Inside the policy node

This is the most-asked-about box in the diagram, so a closer look at the
internals of `crane_policy_node.py`.

### Where the policy comes from

`policy_loader.load_policy(policy_type, checkpoint_path, bounds_min,
bounds_max, cossin, device)` is called once at node startup and returns a
`PolicyBase` instance. Four types are supported:

| `--policy_type` | Class | Checkpoint format | Notes |
|-----------------|-------|-------------------|-------|
| `bc`, `bcrl`, `rl` | `RslRlPolicy` | RSL-RL `.pt` (`model_state_dict` key, or raw state dict) | `_build_model` inspects the keys: if any contain `"encoder"` it builds the `PointNetActorCritic` (3072-dim PCD obs); otherwise a plain RSL-RL `ActorCritic` MLP. `model.eval()` + `torch.no_grad()` for inference. |
| `sac` | `SACPolicy` | Stable-Baselines3 `.zip` | `SAC.load(path)`; `model.predict(obs, deterministic=True)`. |
| `heuristic` | `HeuristicPolicy` | none | Selects the highest valid point in the PCD (`argmax(pc[:, 2])`) and estimates yaw from a 30 cm-radius PCA. Useful as a baseline / sanity check. |

The `PointNetActorCritic` class is loaded by **file path**, not import,
to avoid pulling in the full `crane_testbed` package (and Isaac Lab) on
the system Python interpreter. See `RslRlPolicy._load_pointnet_class`.

### What the policy outputs (and what `decode_action` does)

The model produces a raw 5D action (or 4D if `--cossin false`). It is
squashed and decoded by `policy_loader.decode_action`:

```
a = tanh(action)                                           # ∈ [-1, 1]^5
x = bounds_min[0] + (a[0] + 1) / 2 * (bounds_max[0] - bounds_min[0])
y = bounds_min[1] + (a[1] + 1) / 2 * (bounds_max[1] - bounds_min[1])
z = bounds_min[2] + (a[2] + 1) / 2 * (bounds_max[2] - bounds_min[2])
yaw = atan2(a[4], a[3]) / 2                                # 5D cos/sin
```

The 5D `cos/sin` encoding (with `/ 2`) collapses the basegrapple's 180°
yaw symmetry. `bounds_min` and `bounds_max` start as defaults but get
**overwritten on the first `/crane/action_bounds` message** — see
`_action_bounds_cb`. This matters because the latched bounds the sim
publishes are the exact ones the policy was trained against; if the real
rack is in a different location, the static publisher must publish bounds
that match the deployed checkpoint's training set.

### One grasp cycle, in code

`_tick()` runs at 10 Hz on a ROS2 timer. There are two paths through it:

**A) The "I need a new target" path** — runs once per cycle:

```
if self.waiting_for_target:
    if depth and intrinsics not ready: return
    obs = process_depth(depth, intrinsics, cam_pose, base_pose, ...)  # → 1024×3 PCD in base frame
    x, y, z, yaw = self.policy.get_target(obs)                        # forward pass + decode
    self.fsm.set_target(x, y, z, yaw)                                 # latch the target into the FSM
    self.waiting_for_target = False
```

`waiting_for_target` is a single boolean — the gate that controls when
the next camera frame is consumed and the next policy forward pass
happens. Nothing else is captured; the depth callback keeps overwriting
`self.latest_depth` between cycles, but the policy only reads it when
this flag flips back to `True`.

**B) The FSM tick path** — runs every 10 Hz tick:

```
cmd = self.fsm.tick(current_ee_pos, current_ee_yaw, current_gripper)
joint_yaw = self._grapple_yaw_to_joint(cmd.yaw)   # cached unless cmd.yaw changes
self._send_target_to_jv(cmd.position, joint_yaw, cmd.gripper)  # SetTarget service call
if cmd.cycle_complete:
    self.waiting_for_target = True  # ← arms path A for the next cycle
```

So the contract between the policy node and the FSM is:

- **Policy node → FSM:** `set_target(x, y, z, yaw)` once per cycle.
- **FSM → policy node:** an `EECommand` every tick with `position`, `yaw`,
  `gripper`, `phase`, and `cycle_complete`. The policy node never reads
  the FSM's internal phase to decide what to do — it just forwards the
  command and re-arms the cycle when `cycle_complete` is true.

### When does the next camera frame get consumed?

Only when `cycle_complete` is `True`. That flag is set inside the FSM
itself when it leaves the `CLEAR` phase (the lift-back-up at the end of
deposition) — i.e. **after the log has been picked up, carried to the
trailer, dropped, and the arm has cleared back over the drop zone**.
Until then, the FSM is running purely on its latched target and the
controller is tracking it; new camera frames are dropped on the floor.

This is intentional — the policy is trained on a single grasp decision
per pile interaction, not on closed-loop visual servoing. Re-running the
forward pass at 10 Hz would (a) double-count gripper-occluded views, and
(b) cause the FSM target to wobble mid-grasp.

### Policy node ↔ controller node contract

The only channel from policy node to controller is the `SetTarget`
service. `_send_target_to_jv()` enforces a small but important rule: the
service is only called when the target **changes** by more than 1 mm /
1 mrad / 1 % gripper:

```
new_target = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3),
              round(yaw, 3), round(gripper, 3))
if hasattr(self, '_last_jv_target') and self._last_jv_target == new_target:
    return
```

Without this gate the controller would re-run ikpy on every tick, which
both burns CPU and causes visible joint jitter when IK returns slightly
different solutions for the same target.

The basegrapple-frame `cmd.yaw` from the FSM is converted to a yaw-joint
target via `_grapple_yaw_to_joint()` before the service call. The
conversion picks (out of `{base, base ± π}`) the candidate numerically
closest to `current_yaw_joint`, so the PD controller never takes the long
way around. The result is **cached** on phase change / yaw change so the
joint doesn't chase basegrapple drift mid-descent and visibly spin.

### Inputs the policy node depends on

- `/zedx/depth` + `/zedx/camera_info` — depth image and intrinsics for
  the PCD pipeline.
- `/tf_static` — looked up once on first availability for camera and
  base extrinsics (`_update_transforms_from_tf`).
- `/crane/ee_pos_base`, `/crane/grapple_yaw_base`, `/crane/joint_states`
  — the basegrapple position / yaw / yaw-joint that the FSM and the
  yaw-joint conversion need to decide phase transitions.
- `/crane/action_bounds` (latched, `TRANSIENT_LOCAL`) — the workspace
  AABB the policy was trained against. Subscribed so it arrives once
  on connect and overrides the hardcoded defaults.

### Outputs the policy node produces

- `/crane/grasp_target`, `/crane/policy_target` — the policy's raw grasp
  decision (for visualization / logging).
- `/crane/ee_command` — the FSM's per-tick EE pose (basegrapple frame).
- `/crane/gripper_command` — the FSM's per-tick gripper opening.
- `/crane/fsm_state` — current phase + cycle index as a string.
- `/crane/set_target` (service call) — what the controller actually
  acts on.

## Useful commands

```bash
# Watch the FSM phase transitions
ros2 topic echo /crane/fsm_state

# Watch the EE target the policy is sending to the controller
ros2 topic echo /crane/ee_command

# Watch the joint command being sent to sim/PLC
ros2 topic echo /crane/joint_command

# Inspect the latched action bounds
ros2 topic echo --once /crane/action_bounds
```

The policy node also writes a per-tick CSV to
`/workspace/crane_testbed/scripts/ros2/policy_node_log.csv` with phase, EE
position, target, position error, yaw error, and gripper state.
