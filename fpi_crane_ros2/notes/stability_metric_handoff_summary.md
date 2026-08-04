# Stability Metric Handoff Summary — Real Crane Reward Measurement

## 1. Project context

The goal is to compute and validate a real-world version of the **stability reward metric** used in the BC+RL log-loading crane policy. This metric was one of the most improved reward components in the BC+RL model compared with the pure BC model.

The reward measures **grapple levelness after lift**. In simulation, the stability reward is defined from the angle between the grapple frame z-axis and the world/base up direction:

\[
\varsigma = \left(\max(0, \hat z^g \cdot \hat z^w)\right)^4
\]

where:

- \(\hat z^g\): grapple z-axis
- \(\hat z^w\): world/base up vector
- exponent \(p = 4\) in the paper
- \(\max(0,\cdot)\) prevents an upside-down grapple from receiving a positive reward due to the even exponent

For real-world measurements, the current target frame representing the stability-relevant grapple orientation is:

```text
grapplecarrier
```

and the main parent frame is:

```text
base_link
```

The desired real-world quantity is therefore:

\[
\hat z^g_{base} = R^{base}_{grapplecarrier}[:,2]
\]

and the stability metric becomes:

\[
d = \hat z^g_{base} \cdot \hat z^{ref}_{base}
\]

\[
\theta = \cos^{-1}(\operatorname{clip}(d,-1,1))
\]

\[
\varsigma = \max(0,d)^p
\]

where \(\hat z^{ref}_{base}\) is the chosen "up" reference expressed in `base_link`.

---

## 2. Available data

The user collected several small static rosbag datasets from June 22, 2026:

```text
rosbag2_cranelab_test_2026_06_22-20_31_25/  crane in gaze pose
rosbag2_cranelab_test_2026_06_22-21_01_41/  grasped bundle lifted above rack, bundle in line with stick
rosbag2_cranelab_test_2026_06_22-21_03_58/  grasped bundle lifted above rack, bundle perpendicular to stick
rosbag2_cranelab_test_2026_06_22-21_09_17/  grapple over trailer with grasped bundle
```

There is also a full autonomous grasp rosbag:

```text
crane_run_20260703_161820/
```

with FSM states:

```text
GAZE
HOVER_UP
ALIGN_YAW
DESCEND
CLOSE
LIFT_HIGH
MEASURE
CARRY_HOME
ALIGN_HOME_YAW
LOWER_TO_DROP
OPEN
SETTLE
CLEAR
```

At the time of the discussion, there was **no MEASURE phase implemented**, even though the FSM includes that state. During `LIFT_HIGH`, the crane lifts the grasped bundle above the pile. Most of `LIFT_HIGH` is moving, roughly 45–52 s, with a short mostly static region around 52–54 s before the next target pose.

The core ROS topics discussed:

```text
/joint_states
/tf
/tf_static
/zed_0/zed_node/imu/data
```

The uploaded transform tree showed the dynamic chain:

```text
base_link
  -> mast
  -> mainboom
  -> stick
  -> telescope
  -> upperpassive
  -> lowerpassive
  -> grapplecarrier
```

The passive joints are:

```text
hanger_joint
bearingfork_joint
```

Both are outfitted with encoders and appear in `/joint_states`.

---

## 3. Frame and URDF work

### 3.1 Frame tree

The `view_frames` PDF confirmed that `grapplecarrier` is downstream of:

```text
base_link → mast → mainboom → stick → telescope → upperpassive → lowerpassive → grapplecarrier
```

and that dynamic transforms were being broadcast at roughly 18 Hz.

### 3.2 URDF trimming for IKPy

The full URDF was trimmed to a minimal FK-only URDF containing only the chain needed for:

```text
base_link -> grapplecarrier
```

The minimal chain contains 8 links and 7 joints:

```text
base_link
mast
mainboom
stick
telescope
upperpassive
lowerpassive
grapplecarrier
```

Required joints:

```text
slew_joint
boom_joint
stick_joint
telescope_joint
hanger_joint
bearingfork_joint
grapplecarrier_joint
```

Branches that can be removed from the IKPy FK URDF:

```text
trailer
camera_zed
zed_1_camera_link
mast_sensor_bracket
lidar_0/os_sensor
zed_0_camera_link
grappletong1
grappletong2
grappletong1_tip
grappletong2_tip
```

and their corresponding joints.

For IKPy, visuals, meshes, materials, collisions, inertials, and sensor branches are unnecessary for FK.

### 3.3 Important URDF caveat

The original URDF had:

```xml
<axis rpy="0 -1.22173 0" xyz="0 0 1"/>
```

inside `grapplecarrier_joint`.

Standard URDF parsers generally ignore `rpy` inside `<axis>`. If that axis tilt is physically intended, it should normally be represented in the joint `<origin rpy="...">`, not `<axis rpy="...">`.

However, do not change this yet without checking the intended physical frame convention, because both `robot_state_publisher` and IKPy appear to currently agree using the existing interpretation.

---

## 4. FK service implementation and validation

A ROS 2 FK service was implemented using IKPy and `/joint_states`.

The service call:

```bash
ros2 service call /get_grapple_fk fpi_crane_msgs/srv/GrappleFK "{use_latest_joint_state: true}"
```

returned:

```text
success=True
message='Computed base_link -> grapplecarrier'

pose.position:
x=-4.022621764805785
y= 2.650002930124592
z= 2.5179836028613574

pose.orientation xyzw:
x=-0.00523918150643305
y=-0.018488096442954564
z= 0.8381400075891394
w=-0.5451165645488457

z_axis:
x= 0.011373999980936902
y=-0.0367031558336826
z= 0.9992614825341174
```

A direct TF query gave:

```text
base_link -> grapplecarrier

Translation:
[-4.023, 2.650, 2.518]

Quaternion xyzw:
[0.005, 0.018, -0.838, 0.545]

Rotation matrix:
-0.406  0.914  0.011 -4.023
-0.914 -0.405 -0.037  2.650
-0.029 -0.025  0.999  2.518
 0.000  0.000  0.000  1.000
```

The FK service and TF agree. The quaternion sign difference is not an issue because:

\[
q \equiv -q
\]

The z-axis from FK:

\[
[0.011374,\ -0.036703,\ 0.999261]
\]

matches the third column of the TF rotation matrix.

This validates that:

- IKPy chain construction is correct
- joint name mapping is correct
- passive joints are included
- telescope prismatic joint is handled
- `/joint_states` calibrated values are being used consistently
- TF and FK agree software-wise

It does **not** independently validate physical accuracy of the URDF, encoder offsets, or frame placement.

---

## 5. Stability reference modes

The stability service compares the grapple z-axis against several possible up references.

### 5.1 Mode 1: `base_link_z`

Ignore trailer/base tilt and use:

\[
\hat z^{ref}_{base} = [0,0,1]^T
\]

This computes levelness relative to the crane base frame.

### 5.2 Mode 2: `free_hanging_gaze`

Use the free-hanging unloaded gaze-pose vector as a session-specific gravity estimate.

From the gaze bag:

```text
z_gaze_base =
[0.011373999980936902,
 -0.0367031558336826,
  0.9992614825341174]
```

This assumes the unloaded passive grapple hangs along true gravity. It suggests the trailer/base frame may be tilted around 2 degrees relative to true gravity.

Caveats:

- passive joints may have friction
- hoses/cables may bias the hanging angle
- `grapplecarrier` may not be exactly aligned with the physical plumb axis
- base tilt may change if the trailer/stabilizers/ground/loading changes

### 5.3 Mode 3: `zed_imu_acceleration`

Use the ZED IMU linear acceleration vector as a gravity estimate.

For a static IMU:

\[
\hat z^{imu}_{up} = \frac{a^{imu}}{\|a^{imu}\|}
\]

based on the ZED sign convention observed experimentally.

Then transform to base:

\[
\hat z^{base}_{up} = R^{base}_{imu}\hat z^{imu}_{up}
\]

This mode is only valid when the IMU is quasi-static. During crane motion, acceleration contains gravity plus dynamic acceleration, vibration, and rotational effects:

\[
a_{measured} = a_{gravity} + a_{motion} + a_{vibration} + a_{rotational}
\]

In practice, this mode often failed or produced questionable values during `LIFT_HIGH`.

### 5.4 Mode 4: `zed_imu_orientation`

Use the fused orientation quaternion from `/zed_0/zed_node/imu/data`.

This is currently the best ZED-based mode.

The ZED X appears to have accelerometer + gyroscope, but no magnetometer. That is acceptable for gravity direction because gravity depends on roll/pitch, not yaw.

---

## 6. ZED IMU orientation convention

The user verified whether the orientation quaternion gives:

1. orientation of IMU relative to Earth, \(R^E_I\), or
2. orientation of Earth relative to IMU, \(R^I_E\)

These are transposes of each other:

\[
R^I_E = (R^E_I)^T
\]

Several static tests were run with a spare ZED2i camera on a desk:

- right side up
- upside down
- tilted left
- lens on table

The normalized accelerometer vector was compared against both:

- third row of the rotation matrix
- third column of the rotation matrix

The data strongly showed that the ZED quaternion convention is:

\[
R^E_I
\]

meaning it maps IMU-frame vectors into Earth-frame vectors:

\[
v^E = R^E_I v^I
\]

Therefore, Earth-up expressed in the IMU frame is:

\[
\hat z^I_{up} = (R^E_I)^T [0,0,1]^T
\]

Equivalently, it is the **third row** of \(R^E_I\), not the third column.

Correct code:

```python
from scipy.spatial.transform import Rotation
import numpy as np

def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)

def zed_orientation_gravity_up_base(q_xyzw, R_base_imu):
    R_earth_imu = Rotation.from_quat(q_xyzw).as_matrix()

    # Earth up expressed in IMU frame.
    z_up_imu = R_earth_imu.T @ np.array([0.0, 0.0, 1.0])

    # Same as:
    # z_up_imu = R_earth_imu[2, :]

    z_up_base = R_base_imu @ z_up_imu
    return normalize(z_up_base)
```

---

## 7. ZED left-camera-to-IMU transform

A spare ZED2i showed the static transform:

```text
zed_left_camera_frame -> zed_imu_link

Translation:
[-0.002, -0.023, 0.000]

RPY:
[0.137 deg, 0.026 deg, -0.618 deg]
```

The rotation is close to identity. For preliminary work, using `zed_0_left_camera_frame` as a proxy for `zed_0_imu_link` is acceptable. For final work, the actual frame should be used:

```bash
ros2 run tf2_ros tf2_echo base_link zed_0_imu_link
```

or compose:

\[
R^b_i = R^b_c R^c_i
\]

where:

- \(b\): `base_link`
- \(c\): `zed_0_left_camera_frame`
- \(i\): `zed_0_imu_link`

During one service run, the node was launched with:

```bash
ros2 run fpi_crane_rl_metrics stability_service_node \
  --ros-args -p imu_frame_override:=zed_0_left_camera_frame
```

This was used because `zed_0_imu_link` may not have been available in that bag.

---

## 8. IMU acceleration versus orientation tests

### 8.1 Static desk test

A static ZED2i desk sample gave:

```text
Gravity-up from acceleration:
[-0.01204837, 0.01237277, 0.99985086]

Gravity-up from orientation:
[-0.01901803, 0.01611638, 0.99968924]

Angle difference:
0.453 deg
```

This confirms that acceleration and orientation agree well when the camera is static.

### 8.2 Moving crane data

During a crane grasp, accelerometer and gyro streams were unstable. In one extracted sample:

```text
linear_acceleration:
[-1.5373, 1.0802, 9.6181]

angular_velocity:
[0.0224, 0.3020, 0.1170]
```

Angular speed:

\[
\|\omega\| \approx 0.325\ \mathrm{rad/s} \approx 18.6^\circ/s
\]

This is not static. Acceleration-only gravity is therefore unreliable.

The orientation stream was much more stable. During the final static-ish window of `LIFT_HIGH`, the mean orientation matrix was:

```text
[[ 0.61469771 -0.75588896  0.22534109]
 [ 0.71421228  0.65464211  0.24767826]
 [-0.33473503  0.00869411  0.94227219]]
```

Using the confirmed convention, gravity-up in the IMU frame is the third row:

```text
[-0.33473503, 0.00869411, 0.94227219]
```

After transforming to base, this gave approximately:

```text
z_up_base ≈ [-0.002 to -0.003, 0.051 to 0.053, 0.9986]
```

The corresponding reward was around:

```text
tilt ≈ 8.8 deg
reward ≈ 0.95
```

for one of the service samples.

---

## 9. Implemented stability service behavior

The user implemented `imu_orientation` mode into the stability service.

A service call during the mostly static part of `LIFT_HIGH` was:

```bash
ros2 service call \
  /get_stability \
  fpi_crane_msgs/srv/Stability \
  "{mode: 4, exponent: 4.0}"
```

The service returned four metrics:

```text
base_link_z
free_hanging_gaze
zed_imu_acceleration
zed_imu_orientation
```

### 9.1 First call

```text
stamp:
1783095546.331792000

grapple_z_base:
[-0.0008525, 0.2058250, 0.9785884]
```

Results:

```text
base_link_z:
tilt = 11.8779 deg
reward = 0.9171

free_hanging_gaze:
tilt = 13.9986 deg
reward = 0.8864

zed_imu_acceleration:
invalid
No recent IMU samples passed quasi-static acceleration/angular-rate checks

zed_imu_orientation:
reference_z_base = [-0.00210, 0.05346, 0.99857]
tilt = 8.8136 deg
reward = 0.9536
samples = 16
concentration = 1.0000
```

### 9.2 Second call

```text
stamp:
1783095546.461753000

grapple_z_base:
[-0.0059561, 0.2401135, 0.9707266]
```

Results:

```text
base_link_z:
tilt = 13.8976 deg
reward = 0.8879

free_hanging_gaze:
tilt = 16.0276 deg
reward = 0.8533

zed_imu_acceleration:
invalid

zed_imu_orientation:
reference_z_base = [-0.00182, 0.05309, 0.99859]
tilt = 10.8529 deg
reward = 0.9304
samples = 40
concentration = 1.0000
```

### 9.3 Third call

```text
stamp:
1783095546.931811000

grapple_z_base:
[-0.0169655, 0.2796391, 0.9599553]
```

Results:

```text
base_link_z:
tilt = 16.2694 deg
reward = 0.8492

free_hanging_gaze:
tilt = 18.4157 deg
reward = 0.8104

zed_imu_acceleration:
valid but suspicious
reference_z_base = [0.43366, 0.47546, 0.76543]
tilt = 30.6412 deg
reward = 0.5480
samples = 2
concentration = 0.9976

zed_imu_orientation:
reference_z_base = [-0.00233, 0.05164, 0.99866]
tilt = 13.3065 deg
reward = 0.8969
samples = 14
concentration = 1.0000
```

---

## 10. Interpretation of changing service outputs

The changing reward values are mainly caused by the **grapple z-axis changing**, not the gravity reference.

Across the three calls:

```text
zed_imu_orientation reference_z_base

call 1: [-0.00210, 0.05346, 0.99857]
call 2: [-0.00182, 0.05309, 0.99859]
call 3: [-0.00233, 0.05164, 0.99866]
```

This is very stable, changing by roughly 0.1 deg.

But:

```text
grapple_z_base

call 1: [-0.00085, 0.20583, 0.97859]
call 2: [-0.00596, 0.24011, 0.97073]
call 3: [-0.01697, 0.27964, 0.95996]
```

The first-to-third grapple z-axis change is roughly 4.5 deg in about 0.6 s of bag time.

Therefore, the passive grapple/load is likely still swinging or settling at the end of `LIFT_HIGH`. Instantaneous service calls are valid but not ideal for assigning one scalar reward to the grasp.

The `zed_imu_acceleration` result with only 2 samples should be considered unreliable. The acceleration mode should require more samples and should be rejected if it disagrees significantly with orientation-derived gravity.

---

## 11. Important ROS time issue

At one point, the IMU acceleration mode failed with:

```text
Latest IMU sample is stale: age=511453.733s > 0.500s
```

This likely came from a ROS time mismatch.

For rosbag playback, use:

```bash
ros2 bag play crane_run_20260703_161820/ --clock --rate 0.1
```

and launch the service with:

```bash
--ros-args -p use_sim_time:=true
```

If `use_sim_time` is not set, the node compares bag timestamps against wall time and may think IMU samples are stale.

---

## 12. The `max()` and `clip()` distinction

The reward is:

\[
\varsigma = \max(0,d)^4
\]

where:

\[
d = \hat z_g \cdot \hat z_{ref}
\]

The `max(0,d)` is a **reward-shaping guard**. Without it, because the exponent is even:

\[
(-1)^4 = 1
\]

so a perfectly upside-down grapple would receive maximum reward. The max ensures that any angle greater than 90 degrees gets zero reward.

The `clip(d,-1,1)` used before `acos` is only a **numerical safety guard**, preventing floating-point values such as 1.0000000002 from causing an invalid arccos.

Correct implementation:

```python
d_raw = np.dot(z_grapple, z_ref)
d = np.clip(d_raw, -1.0, 1.0)

tilt_rad = np.arccos(d)
tilt_deg = np.rad2deg(tilt_rad)

reward = max(0.0, d) ** exponent
```

---

## 13. Current recommendation

### 13.1 Main metric modes to keep

Keep these as primary comparison modes:

```text
base_link_z
free_hanging_gaze
zed_imu_orientation
```

### 13.2 Demote or remove accelerometer-only mode

The accelerometer-only mode should not be the primary mode during `LIFT_HIGH`, because the mast-mounted ZED is moving/vibrating.

If kept, it should be a debug/static-calibration mode with stricter rules:

```text
minimum samples >= 20 or 50
minimum averaging window >= 0.5 s
acceleration norm close to 9.81 m/s²
angular speed below threshold
accel-vs-orientation gravity disagreement < 2–3 deg
```

The service should not allow a valid accelerometer reward from only 2 samples.

### 13.3 Prefer orientation-derived gravity

The ZED orientation stream is currently the best ZED-based estimate of gravity direction because:

- it is stable in the crane data
- it agrees with accelerometer gravity in static desk tests
- it is less sensitive to instantaneous translational acceleration
- lack of magnetometer mainly affects yaw, which is not important for gravity/up

### 13.4 Use `zed_0_imu_link` if available

The current override:

```text
imu_frame_override:=zed_0_left_camera_frame
```

is acceptable for quick tests, but final analysis should use the real IMU frame:

```text
zed_0_imu_link
```

or explicitly compose:

```text
base_link -> zed_0_left_camera_frame -> zed_0_imu_link
```

---

## 14. Recommended next steps

### Step 1: Move from instantaneous service calls to offline bag analysis

The service is useful for live debugging, but the real reward should be computed offline over a defined window.

Create a script that reads:

```text
/tf
/joint_states
/zed_0/zed_node/imu/data
/crane/fsm_state or equivalent FSM topic
```

and outputs a time series:

```text
time
fsm_state
grapple_z_base
base_link_z_tilt_deg
base_link_z_reward
free_hanging_gaze_tilt_deg
free_hanging_gaze_reward
zed_imu_orientation_tilt_deg
zed_imu_orientation_reward
zed_imu_acceleration_tilt_deg, if valid
zed_imu_acceleration_reward, if valid
```

### Step 2: Define an evaluation window

Do not use a random single service call as the reward.

Use something like:

```text
last 1.0 s of LIFT_HIGH
```

or, better:

```text
future MEASURE phase: 1–2 s hold after lift
```

Then report:

```text
mean reward
median reward
min reward
final reward
mean tilt
max tilt
standard deviation
```

### Step 3: Implement or activate the MEASURE phase

The most defensible experimental procedure is:

```text
LIFT_HIGH
hold still for 1–2 seconds
MEASURE
continue to CARRY_HOME
```

Compute the stability metric during `MEASURE`.

### Step 4: Add query-time/window support to the service

Extend the service interface with fields like:

```srv
builtin_interfaces/Time query_time
float64 window_sec
bool use_latest
```

Then the service can compute either:

```text
instantaneous stability at query_time
```

or:

```text
windowed stability around query_time
```

This would make the service more useful for repeatable measurements and less dependent on when the user hits Enter.

### Step 5: Plot the reward traces

Plot over the entire `LIFT_HIGH` phase:

```text
grapple_z_base components
tilt_deg_base
tilt_deg_gaze
tilt_deg_imu_orientation
reward_base
reward_gaze
reward_imu_orientation
fsm_state
```

This will show whether the perceived instability is real passive settling and help identify the correct measurement window.

### Step 6: Validate the visual frame

In RViz, display the `grapplecarrier` axes and confirm that its positive z-axis corresponds to the paper’s intended grapple \(\hat z^g\).

The visual impression of a 30-degree tilt may come from:

- the tongs
- the log bundle
- camera perspective
- a different grapple sub-frame
- or observing a different timestamp

while the metric uses only `grapplecarrier` z.

### Step 7: Compare to sim

Once a robust real-world time-window metric is defined, export a matching sim trajectory and compute the same quantities:

```text
sim_grapple_z_world
real_grapple_z_base
real_gravity_reference_base
tilt_deg
reward_p4
reward_p2 / p6 sensitivity
```

Do not tune the exponent until the physical angle measurement is trusted.

---

## 15. Useful formulas

### Extract grapple z from TF

Given:

\[
T^b_g =
\begin{bmatrix}
R^b_g & p^b_g\\
0 & 1
\end{bmatrix}
\]

the grapple z-axis in base is:

\[
\hat z^b_g = R^b_g[:,2]
\]

### Base-relative reward

\[
\hat z^b_{ref} = [0,0,1]^T
\]

\[
d = \hat z^b_g \cdot \hat z^b_{ref}
\]

\[
r = \max(0,d)^p
\]

### Gaze-calibrated reward

\[
\hat z^b_{ref} =
[0.011374,\ -0.036703,\ 0.999261]^T
\]

\[
d = \hat z^b_g \cdot \hat z^b_{ref}
\]

\[
r = \max(0,d)^p
\]

### ZED orientation gravity

Given ZED quaternion \(q\) and \(R^b_i\):

\[
R^E_I = R(q)
\]

\[
\hat z^I_{up} = (R^E_I)^T[0,0,1]^T
\]

\[
\hat z^b_{up} = R^b_i \hat z^I_{up}
\]

Then:

\[
d = \hat z^b_g \cdot \hat z^b_{up}
\]

\[
r = \max(0,d)^p
\]

### Orientation-derived gravity in code

```python
from scipy.spatial.transform import Rotation
import numpy as np

def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)

def gravity_up_from_zed_orientation(q_xyzw, R_base_imu):
    R_earth_imu = Rotation.from_quat(q_xyzw).as_matrix()
    z_up_imu = R_earth_imu.T @ np.array([0.0, 0.0, 1.0])
    z_up_base = R_base_imu @ z_up_imu
    return normalize(z_up_base)
```

### Windowed average of gravity vectors

Instead of averaging rotation matrices directly, average the derived unit gravity vectors:

```python
z_samples = []

for imu_msg in imu_window:
    q_xyzw = [
        imu_msg.orientation.x,
        imu_msg.orientation.y,
        imu_msg.orientation.z,
        imu_msg.orientation.w,
    ]

    R_base_imu = lookup_rotation(
        "base_link",
        imu_msg.header.frame_id,
        imu_msg.header.stamp,
    )

    z_up_base = gravity_up_from_zed_orientation(q_xyzw, R_base_imu)
    z_samples.append(z_up_base)

z_mean_raw = np.mean(z_samples, axis=0)
concentration = np.linalg.norm(z_mean_raw)
z_mean = z_mean_raw / concentration
```

A concentration close to 1 means the vector estimates agree well.

---

## 16. Current status

Completed:

- Identified `grapplecarrier` as stability frame.
- Verified the TF chain and minimal URDF chain.
- Built an IKPy FK service.
- Verified IKPy FK agrees with TF.
- Clarified stability reward math.
- Estimated trailer/base tilt from free-hanging gaze pose.
- Investigated ZED IMU acceleration vs orientation.
- Confirmed ZED quaternion convention using static tests.
- Implemented `imu_orientation` mode in the stability service.
- Demonstrated that IMU orientation gravity is stable during the static-ish end of `LIFT_HIGH`.
- Observed that instantaneous rewards change significantly because the grapple is still settling.

Main unresolved issues:

- Need final use of `zed_0_imu_link` instead of left-camera override.
- Need offline bag analysis over a defined window.
- Need better FSM-defined `MEASURE` phase or post-lift hold.
- Need stricter accelerometer validity checks or removal of accelerometer-only mode.
- Need RViz/frame validation to ensure `grapplecarrier` z visually corresponds to intended stability axis.
- Need sim trajectory comparison using the same metric implementation.
