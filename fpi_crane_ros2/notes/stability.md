# Stability Reward Metric

In simulation the stability reward denoted the grapple levelness after lift. Quantitatively, this is the angle between grapple $ẑ^g$ and base-frame $ẑ^w$ (world frame), 

$\varsigma=(\text{max}(0, {ẑ^g}^Tẑ^w))^4$ 

let $d={ẑ^g}^Tẑ^w$

since both vectors are unit vecotrs the dot product is equivalent to the cosine of the angle between them:

| Angle $\theta$ | Dot product $d=\cos\theta$ | Physical meaning              |
| -------------: | -------------------------: | ----------------------------- |
|      $0^\circ$ |                      1.0 | perfectly upright             |
|     $30^\circ$ |                    0.866 | moderately tilted             |
|     $60^\circ$ |                      0.5 | badly tilted                  |
|     $90^\circ$ |                      0.0 | sideways                      |
|    $120^\circ$ |                     -0.5 | more upside-down than upright |
|    $180^\circ$ |                     -1.0 | perfectly upside-down         |

The $\text{max}()$ function acts as a reward shaping tool to restrict reward to positive dot products: 
> Only reward the component of the grapple z-axis that points in the same upward hemisphere as world/base z. If the grapple z-axis points sideways or downward, give zero stability reward. 

With the positive exponent in the reward structure, upright and upside-down duals will recieve the same reward without the $\text{max}()$ function. 

define angle between vectors as:

$\theta=\text{cos}^{-1}(\text{clip}(d,−1,1))$

where clip is to avoid floating point errors in the dot product

```python
d_for_angle = np.clip(d, -1.0, 1.0)
theta = np.arccos(d_for_angle)
```

The reward can equivalently be expressed in angle form:

$\varsigma=(\text{max}(0, \cos\theta))^4$ 

where $0^\circ\leq\theta\leq 180^\circ$

## Gravity Correction

`base_link` is mounted on a trailer that may be pitched or rolled a few degrees, then:

$ẑ^{base}\neq ẑ^{gravity}$

if the reward is computed with:

$d={ẑ^g}^T\cdot \begin{bmatrix} 0 & 0 & 1 \end{bmatrix}^T$

then what is being measured is grapple levelness relative to the tilted crane base, not relative to gravity. A stronger formulation is:

$d=ẑ^g_{base} \cdot ẑ^{gravity}_{base}$

where $ẑ^{gravity}_{base}$ is the gravity-up vector expressed in the crane base frame.

the error is only a constant offset if the trailer/base tilt is constant. If the trailer suspension or crane base flexes during a loaded lift, the base tilt may change slightly during the grasp. That would make it a time-varying disturbance, not just a constant correction.

## Method

### 1. ROS2 transform 

```bash
source /opt/ros/humble/setup.bash
source ~/FPI_liebherr_automation/ros2_ws/install/setup.bash

# echo transform
ros2 run tf2_ros tf2_echo base_link grapplecarrier
```

Echoing the transform using tf2 prints the homogeneous transform matrix:

$$
T^{base}_{grapple} =
\begin{bmatrix}
R^{base}_{grapple} & p^{base}_{grapple} \\
0 & 1
\end{bmatrix}
$$

The grapple z-axis expressed in base_link is the third column of the rotation matrix

$$
ẑ^g_{base}​=R^{base}_{grapple}
\begin{bmatrix}​
​0\\0\\1
\end{bmatrix} = R^{base}_{grapple}​[:,2]
​$$
​

where, $R^{base}_{grapple}$ is the `grapplecarrier` frame orientation expressed in `base_link` frame. $p^{base}_{grapple}$ is the `grapplecarrier` origin expressed in the `base_link` frame. 

The transform maps a point written in `grapplecarrier` coordinates into `base_link` coordinates. Make the point a homogeneous point by adding another row with a unit element. Multiplying the homogeneous point by the transform matrix will give a homegeneous point in `base_link`. Intuitively, this operation is:

> rotate the point from the grapple frame into the base frame, then translate it to the grapple frame’s origin position in the base frame.

The homogeneous transform matrix lets you combine translation and rotation into one object. Instead of an affine expression to describe the complete transform. This also allows for chaining transforms through matrix multiplication (underlying FK).

Generally, many useful linear algebra tools come out of this fomulation of the transformation;
- chaining multiplations
- transform inversion $R^{-1}=R^T$


Internally, ROS stores `transformStamped` messages as a translation and rotation quaternion:

    translation: x, y, z
    rotation: quaternion x, y, z, w

The rotation matrix can be created from the quaternion representation as follows:

```python
def quat_to_R_xyzw(x, y, z, w):
    q = np.array([x, y, z, w], dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)],
    ])
```

### 2. FK on /joint_states 

    joint_states(t) 
        -> FK using URDF 
        -> T_base_grapple(t)
        -> z_g_base(t)
        -> dot(t)
        -> theta(t)
        -> reward(t)

## Uncertainty

### 1. Encoder zero offsets

The URDF joint angle $q=0$ must mean the same physical pose as the encoder joint-state $q=0$. If `hanger_joint` or `bearingfork_joint` has an offset, the computed grapple z-axis will be wrong even if the encoder values are repeatable.

Validate the encoder angles in a pose where the grapple orientation is approximately known.
- IMU on grapple
- Assume gravity will position `grapple_carrier` z in line with gravity.

Check Isaac Sim joint-states in different poses. Compare with rosbag data

### 2. Frame mismatch between sim and real

### 3. Timing: when is measurement taken?

### 4. Timestamp Synchronization

### 5. Reward exponent tuning

Te exponent changes sensitivity but does not change the underlying measurement.


## ROS bags
Open rosbag and publsh time from recording
```bash
ros2 bag play rosbag2_cranelab_test_2026_06_22-20_31_25/ --loop --clock
```
Open rviz2 node to visualize rosbag. Use time published by rosbag rather than PC system time. 
```bash
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true
```


### 06/22/2026 Bags
- `rosbag2_cranelab_test_2026_06_22-20_27_06/`: only joint states and transforms. No camera data
- `rosbag2_cranelab_test_2026_06_22-20_31_25/`: crane in gaze pose
- `rosbag2_cranelab_test_2026_06_22-21_01_41/`: grasped bundle lifted above rack. Bundle in line with stick
- `rosbag2_cranelab_test_2026_06_22-21_03_58/`: grasped bundle lifted above rack. Bundle perpendicular to stick
- `rosbag2_cranelab_test_2026_06_22-21_09_17/`: grapple over trailer with grasped bundle
- `rosbag2_cranelab_test_2026_06_22-21_10_25/`: empty
- `rosbag2_cranelab_test_2026_06_22-21_11_02/`: empty
- `rosbag2_cranelab_test_2026_06_22-21_15_34/`: empty
- `rosbag2_cranelab_test_2026_06_22-21_16_50/`: empty

#### 20_31_25 - crane in gaze pose
Transform

    At time 1782160289.189288119
    - Translation: [-4.023, 2.650, 2.518]
    - Rotation: in Quaternion (xyzw) [0.005, 0.018, -0.838, 0.545]
    - Rotation: in RPY (radian) [-0.025, 0.029, -1.989]
    - Rotation: in RPY (degree) [-1.449, 1.658, -113.942]
    - Matrix:
    -0.406  0.914  0.011 -4.023
    -0.914 -0.405 -0.037  2.650
    -0.029 -0.025  0.999  2.518
    0.000  0.000  0.000  1.000

/joint_states topic

    header:
    stamp:
        sec: 1782160296
        nanosec: 449316710
    frame_id: ''
    name:
    - slew_joint
    - boom_joint
    - stick_joint
    - telescope_joint
    - hanger_joint
    - bearingfork_joint
    - grapplecarrier_joint
    - grappletong1_joint
    - grappletong2_joint
    position:
    - 0.98960085
    - 0.6475166055555555
    - -0.7557269277777776
    - 0.2814
    - 0.13788089444444443
    - -0.024434588888888886
    - 3.305650811111111
    - 0.0
    - 0.0
    velocity:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0



#### 21_01_41 - grasped bundle lifted above rack. Bundle in line with stick

Transform

    At time 1782162109.791529838
    - Translation: [-4.435, 0.352, 2.542]
    - Rotation: in Quaternion (xyzw) [-0.020, 0.090, -0.764, 0.638]
    - Rotation: in RPY (radian) [-0.165, 0.084, -1.757]
    - Rotation: in RPY (degree) [-9.457, 4.834, -100.652]
    - Matrix:
    -0.184  0.972  0.146 -4.435
    -0.979 -0.169 -0.112  0.352
    -0.084 -0.164  0.983  2.542
    0.000  0.000  0.000  1.000

/joint_states topic

    header:
    stamp:
        sec: 1782162113
        nanosec: 51249736
    frame_id: ''
    name:
    - slew_joint
    - boom_joint
    - stick_joint
    - telescope_joint
    - hanger_joint
    - bearingfork_joint
    - grapplecarrier_joint
    - grappletong1_joint
    - grappletong2_joint
    position:
    - 1.4974912333333332
    - 0.823794711111111
    - -1.0890845333333332
    - 0.2925
    - 0.4223693222222222
    - -0.1012290111111111
    - 3.0438516444444446
    - 0.0
    - 0.0
    velocity:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0

#### 21_03_58

Transform 

    At time 1782162248.281285643
    - Translation: [-4.415, 0.357, 2.516]
    - Rotation: in Quaternion (xyzw) [0.040, -0.066, 0.997, 0.016]
    - Rotation: in RPY (radian) [-0.131, -0.081, 3.115]
    - Rotation: in RPY (degree) [-7.526, -4.652, 178.487]
    - Matrix:
    -0.996 -0.037  0.077 -4.415
    0.026 -0.991 -0.133  0.357
    0.081 -0.131  0.988  2.516
    0.000  0.000  0.000  1.000

/joint_states topic

    header:
    stamp:
        sec: 1782162246
        nanosec: 281209437
    frame_id: ''
    name:
    - slew_joint
    - boom_joint
    - stick_joint
    - telescope_joint
    - hanger_joint
    - bearingfork_joint
    - grapplecarrier_joint
    - grappletong1_joint
    - grappletong2_joint
    position:
    - 1.4974912333333332
    - 0.8185587277777777
    - -1.0890845333333332
    - 0.2931
    - 0.35779219444444443
    - -0.12740892777777776
    - 1.6179188499999997
    - 0.0
    - 0.0
    velocity:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0

## ZED Stereo Camera IMU

IMU in the stereo cameras on the setup, ZED0 and ZED1 can be used to get the world frame z vector. 

### Topics of interest

/zed/zed_node/imu/data` could be used be get the orientation of either camera. These topics differ slightly in their measurement. 

### World ẑ from camera ẑ?

`linear_acceleration` within the imu topic gives the proportion of the acceleration in each direction. These proportions and `orientation` can give world gravity 

    header:
    stamp:
        sec: 1782744708
        nanosec: 582904954
    frame_id: zed_imu_link
    orientation:
    x: -0.002523012226447463
    y: -0.002501532668247819
    z: 0.2794196903705597
    w: 0.9601625800132751
    orientation_covariance:
    - 5.000377401614871e-10
    - -2.817297431891566e-10
    - -3.5338326780228905e-11
    - -2.8172967392717044e-10
    - 1.2870222094639947e-09
    - 2.869760962231486e-10
    - -3.533831379360651e-11
    - 2.869760615921556e-10
    - 5.450157120861446e-10
    angular_velocity:
    x: -0.0006225676272780374
    y: -0.00030953822722953374
    z: 0.0010519925470589661
    angular_velocity_covariance:
    - 1.9168053818630833e-06
    - 0.0
    - 0.0
    - 0.0
    - 1.975252515583047e-06
    - 0.0
    - 0.0
    - -0.0
    - 1.8263128172007136e-06
    linear_acceleration:
    x: 0.03939244523644447
    y: -0.05310821533203125
    z: 9.810694694519043
    linear_acceleration_covariance:
    - 8.318982145283371e-05
    - 0.0
    - 0.0
    - 0.0
    - 9.144768409896642e-05
    - 0.0
    - 0.0
    - -0.0
    - 0.00022149771393742412
    ---


For camera (i):

$$
a_i^{I_i}
=

\begin{bmatrix}
a_{x,i} \\
a_{y,i} \\
a_{z,i}
\end{bmatrix}
$$

Normalize:

$$
\hat a_i^{I_i}
=

\frac{a_i^{I_i}}{|a_i^{I_i}|}
$$

Choose the sign so it represents up:

$$
\hat z^{I_i}_{up}
=

s_i \hat a_i^{I_i}
$$

> example output is with camera upright. ZED_0 on the crane setup is upright. ZED_1 on the crane setup is upside-down

Transform to base:

$$
\hat z^{base}_{up,i}
=

R^{base}_{I_i}
\hat z^{I_i}_{up}
$$

Then average the two estimates:

$$
\hat z^{base}_{up}
=

\frac{
\hat z^{base}_{up,1}
+
\hat z^{base}_{up,2}
}{
\left|
\hat z^{base}_{up,1}
+
\hat z^{base}_{up,2}
\right|
}
$$

> This method inherits the uncertainty from the transforms to the camera IMU.  

Both cameras on the crane hardware have a constant vibration from the engine. Taking a windowed measurement during the data collection phase could wash out the vibration variance; However, sway skew the measurement (especially in the stick camera). In theory, if there is perfect time alignment with the /tf and /zed/imu/data data streams this would not be a huge issue. 

The imu could also be used to determine a good time to take the measurement. Quantify how long the decaying oscillations last for and when they are negligable. The sway duration could be added to the reward framework in future implementations. 

### IMU topic bug on crane hardware

imu topic is not publishing properly.

    header:
    stamp:
        sec: 0
        nanosec: 0
    frame_id: zed_1_imu_link
    orientation:
    x: 0.0
    y: 0.0
    z: 1.0
    w: 0.0
    orientation_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    angular_velocity:
    x: 0.0
    y: 0.0
    z: 0.0
    angular_velocity_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    linear_acceleration:
    x: 0.0
    y: 0.0
    z: 0.0
    linear_acceleration_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    ---

    header:
    stamp:
        sec: 0
        nanosec: 0
    frame_id: zed_0_imu_link
    orientation:
    x: 0.0
    y: 1.0
    z: 0.0
    w: 0.0
    orientation_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    angular_velocity:
    x: 0.0
    y: 0.0
    z: 0.0
    angular_velocity_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    linear_acceleration:
    x: 0.0
    y: 0.0
    z: 0.0
    linear_acceleration_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    ---

potentially could be a result of this parameter: `publish_imu_tf = LaunchConfiguration('publish_imu_tf'`

could also be related to the PC running the preception_kit launch file. Communication bandwidth not large enough to accomodate transmission to another machine. 

limitations depending on SDK version, or DDS 

## 07/03/2026 - FK with ikpy vs ROS2 tf2

Generate a URDF file from the xacro of the crane:

```bash
ros2 run xacro xacro \
  /path/to/fpi_crane.urdf.xacro \
  -o ~/FPI_liebherr_automation/ros2_ws/src/fpi_crane_ros2/fpi_crane_kinematics/urdf/fpi_crane.urdf
```
Trim the urdf to the necessary links and joints to avoid ikpy errors and make debuging easier.

```bash
ros2 service call /get_grapple_fk fpi_crane_msgs/srv/GrappleFK "{use_latest_joint_state: true}"
```

    requester: making request: fpi_crane_msgs.srv.GrappleFK_Request(use_latest_joint_state=True, joint_state=sensor_msgs.msg.JointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id=''), name=[], position=[], velocity=[], effort=[]))

    response:
    fpi_crane_msgs.srv.GrappleFK_Response(
    - success=True, 
    - message='Computed base_link -> grapplecarrier',
    - pose=geometry_msgs.msg.PoseStamped(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=1782160291, nanosec=409238171), frame_id='base_link'), 
    - pose=geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=-4.022621764805785, y=2.650002930124592, z=2.5179836028613574), 
    - orientation=geometry_msgs.msg.Quaternion(x=-0.00523918150643305, y=-0.018488096442954564, z=0.8381400075891394, w=-0.5451165645488457))), 
    - z_axis=geometry_msgs.msg.Vector3(x=0.011373999980936902, y=-0.0367031558336826, z=0.9992614825341174),
    - joint_state_used=sensor_msgs.msg.JointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=1782160291, 
    - nanosec=409238171), 
    - frame_id=''), 
    - name=['slew_joint', 'boom_joint', 'stick_joint', 'telescope_joint', 'hanger_joint', 'bearingfork_joint', 'grapplecarrier_joint', 'grappletong1_joint', 'grappletong2_joint'], 
    - position=[0.98960085, 0.6475166055555555, -0.7557269277777776, 0.2814, 0.13788089444444443, -0.024434588888888886, 3.305650811111111, 0.0, 0.0],
    - velocity=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
    - effort=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

```bash
ros2 run tf2_ros tf2_echo base_link grapplecarrier
```
    [INFO] [1783095947.508160271] [tf2_echo]: Waiting for transform base_link ->  grapplecarrier: Invalid frame ID "base_link" passed to canTransform argument target_frame - frame does not exist
    At time 1782160289.79411668
    - Translation: [-4.023, 2.650, 2.518]
    - Rotation: in Quaternion (xyzw) [0.005, 0.018, -0.838, 0.545]
    - Rotation: in RPY (radian) [-0.025, 0.029, -1.989]
    - Rotation: in RPY (degree) [-1.449, 1.658, -113.942]
    - Matrix:
    -0.406  0.914  0.011 -4.023
    -0.914 -0.405 -0.037  2.650
    -0.029 -0.025  0.999  2.518
    0.000  0.000  0.000  1.000

comparing the ẑ vectors with the angle between them

```python
import numpy as np

z_fk = np.array([
    0.011373999980936902,
    -0.0367031558336826,
    0.9992614825341174,
])

z_tf = np.array([
    0.011,
    -0.037,
    0.999,
])

z_fk /= np.linalg.norm(z_fk)
z_tf /= np.linalg.norm(z_tf)

dot = np.clip(np.dot(z_fk, z_tf), -1.0, 1.0)
angle_error_deg = np.degrees(np.arccos(dot))

print(f"Z-axis disagreement: {angle_error_deg:.6f} deg")
```

Z-axis disagreement: 0.027576 deg

`tf2` and ikpy + `/joint_states` generally agree with each other. Easy to implement Buffer and TransformListener. 

FK service may still be useful for:
- computing grapple pose from a supplied historical JointState
- evaluate hypothetical joint configurations
- returns the exact joint state used

## 07/07/2026 - Stability service with ZED IMU gravtity direction

The ZED ros2 and ZED IMU documentation:

- Accelerometer, gyroscope, and orientation data in Earth frame.
- When an accelerometer is static, it is still measuring an acceleration of 9.8 m/s², which corresponds to the force applied by the Earth’s gravity. This force always has the same direction, from the camera toward the center of the Earth. Gravity allows us to compute the camera’s absolute inclination and detect events like free falls.
- A gyroscope measures the angular velocity of the camera in degrees per second (deg/s). When combined with the accelerometer, both sensors can estimate the orientation of the camera at a high frequency.

If the quaternion in the /zed_0/zed_node/imu/data represents the IMU frame to a gravity-aligned Earth frame, then gravity-up in the IMU frame is:

$$
ẑ_{up}^{IMU}=(R_IMU^{Earth})^T 
\begin{bmatrix}​
​0\\0\\1
\end{bmatrix} 
$$


$$
ẑ_{up}^{base}=R_{IMU}^{Earth} ẑ_{base}^{IMU}
$$

## 07/08/2026

Orientation data stream from the imu versus the linear acceleration stream. The orientation stream is likely computed through sensor fusion of the accelerometer, gyro, and likely magnetometer (I am not certain that there is one) in the zed camera imu. 

imu topic stream from a zed camera:

~$ ros2 topic echo /zed_0/zed_node/imu/data --once

    header:
    stamp:
        sec: 1783095547
        nanosec: 266795000
    frame_id: zed_0_imu_link
    orientation:
    x: -0.06663110852241516
    y: 0.15660500526428223
    z: 0.4092191159725189
    w: 0.8964233994483948
    orientation_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    angular_velocity:
    x: 0.022386412064462426
    y: 0.3019668561410713
    z: 0.11702248898774101
    angular_velocity_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    linear_acceleration:
    x: -1.537304401397705
    y: 1.080188512802124
    z: 9.61812973022461
    linear_acceleration_covariance:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    ---

This imu reading was taken in the measurement phase of the policy fsm (GAZE, HOVER_UP, ALIGN_YAW, DESCEND, CLOSE, LIFT_HIGH, MEASURE, CARRY_HOME, ALIGN_HOME_YAW, LOWER_TO_DROP, OPEN, SETTLE, CLEAR). In the measurement phase the crane lifts the grasped bundle and waits a brief period to let movement stabilize.

The grapple_carrier to base_link for measure phase is:

~$ ros2 run tf2_ros tf2_echo base_link grapplecarrier

At time 1783095553.211334609
- Translation: [-4.783, -0.785, 1.704]
- Rotation: in Quaternion (xyzw) [0.197, 0.143, 0.678, 0.693]
- Rotation: in RPY (radian) [0.487, -0.068, 1.532]
- Rotation: in RPY (degree) [27.914, -3.910, 87.779]
- Matrix:
  0.039 -0.884  0.465 -4.783
  0.997  0.002 -0.078 -0.785
  0.068  0.467  0.882  1.704
  0.000  0.000  0.000  1.000

The base_link to zed_0_left_camera_frame in this measure phase is:

~$ ros2 run tf2_ros tf2_echo base_link zed_0_left_camera_frame

At time 1783095548.665736000
- Translation: [-0.075, -0.218, 1.361]
- Rotation: in Quaternion (xyzw) [-0.164, -0.002, 0.967, -0.192]
- Rotation: in RPY (radian) [0.062, 0.325, -2.739]
- Rotation: in RPY (degree) [3.567, 18.593, -156.945]
- Matrix:
 -0.872  0.373 -0.317 -0.075
 -0.371 -0.926 -0.067 -0.218
 -0.319  0.059  0.946  1.361
  0.000  0.000  0.000  1.000


```python
DeclareLaunchArgument(
    'publish_imu_tf',
    default_value='false',
    description='Enable publication of the IMU TF. Note: Ignored if `publish_tf` is False.',
    choices=['true', 'false']),
```

`publish_imu_tf` needs to be set to `true` in order for the IMU to be placed in the transform tree. 

On similar ZED2i camera, constant left camera to imu transform is as follows:

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

```bash
ros2 run tf2_ros tf2_echo zed_left_camera_frame zed_imu_link
```

    At time 1783536822.138282290
    - Translation: [-0.002, -0.023, 0.000]
    - Rotation: in Quaternion (xyzw) [0.001, 0.000, -0.005, 1.000]
    - Rotation: in RPY (radian) [0.002, 0.000, -0.011]
    - Rotation: in RPY (degree) [0.137, 0.026, -0.618]
    - Matrix:
    1.000  0.011  0.000 -0.002
    -0.011  1.000 -0.002 -0.023
    -0.000  0.002  1.000  0.000
    0.000  0.000  0.000  1.000


for static zed2i camera gravity up computed from accelerometer vs orientation (gyro, accelerometer fusion) generally agree:

    Gravity-up from acceleration: [-0.01204837  0.01237277  0.99985086]
    Gravity-up from orientation: [-0.01901803  0.01611638  0.99968924]
    Angle difference between the two estimates (radians): 0.007913106236056735
    Angle difference between the two estimates (degrees): 0.4533875901647034

Averging data streams over window would likely increase similarity. 

## Initial Service Implementation

Call service

```bash
ros2 service call \
  /get_stability \
  fpi_crane_msgs/srv/Stability \
  "{mode: 4, exponent: 4.0}"
```

    response:
    fpi_crane_msgs.srv.Stability_Response(success=True, message='Computed 2/3 requested stability metrics using exponent 4.0', stamp=builtin_interfaces.msg.Time(sec=1783095547, nanosec=336748000), grapple_z_base=geometry_msgs.msg.Vector3(x=-0.010859820191130978, y=0.20290098321952796, z=0.9791390377847087), metrics=
    
    [fpi_crane_msgs.msg.StabilityMetric(reference_name='base_link_z', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=1.0), dot_product=0.9791390377847087, tilt_rad=0.20461621820852088, tilt_deg=11.723645723276153, reward=0.9191311059147002, samples_used=0), 
    
    fpi_crane_msgs.msg.StabilityMetric(reference_name='free_hanging_gaze', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.011373999980936902, y=-0.0367031558336826, z=0.9992614825341174), dot_product=0.9708453005032165, tilt_rad=0.24206394777138204, tilt_deg=13.869242579575381, reward=0.8883827779529634, samples_used=0), 
    
    fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_acceleration', valid=False, status='Latest IMU sample is stale: age=511453.733s > 0.500s', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), dot_product=0.0, tilt_rad=0.0, tilt_deg=0.0, reward=0.0, samples_used=0)])

IMU timestamp is not working as expected. Does not function properly when using rosbags. 

The orientation message from the imu topic is much more stable than the raw output of the accelerometer and the gyro. Over the duration of the LIFT_HIGH phase the mean orientation is expressed as a rotionation natrix is:

    Mean rotation matrix from quaternions:
    [[ 0.62635358 -0.74495504  0.22961531]
    [ 0.70403774  0.66705343  0.24366081]
    [-0.33468202  0.00904002  0.94228776]]
    Estimated gravity direction in IMU frame: [-0.33468202  0.00904002  0.94228776]

The brief final static moment of the LIFT_HIGH phase (52.7s - 52.2s) is:

    Mean rotation matrix from quaternions:
    [[ 0.61469771 -0.75588896  0.22534109]
    [ 0.71421228  0.65464211  0.24767826]
    [-0.33473503  0.00869411  0.94227219]]
    Estimated gravity direction in IMU frame: [-0.33473503  0.00869411  0.94227219]

This gives a reward of:

    IMU-orientation gravity reward ≈ 0.95
    tilt angle ≈ 8.8 deg


### 07/10/2026 ZED IMU frame convention

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

right side up

    orientation:
    x: -0.003003164194524288
    y: 0.006323532667011023
    z: 0.02891603298485279
    w: 0.9995573163032532
    linear_acceleration:
    x: -0.11213520914316177
    y: -0.07015224546194077
    z: 9.828349113464355

upside down

    orientation:
    x: 0.9984282851219177
    y: -0.05016481876373291
    z: -0.024666011333465576
    w: -0.004019455052912235
    linear_acceleration:
    x: -0.4912603795528412
    y: -0.05330243706703186
    z: -9.73828411102295

tilted left

    orientation:
    x: -0.2726678252220154
    y: 0.006390404887497425
    z: 0.03964107856154442
    w: 0.9612700939178467
    linear_acceleration:
    x: -0.3305545747280121
    y: -5.132851600646973
    z: 8.348133087158203

lens on table

    orientation:
    x: -0.04270610585808754
    y: 0.7230478525161743
    z: 0.02867220900952816
    w: 0.6888801455497742
    linear_acceleration:
    x: -9.794276237487793
    y: -0.17778460681438446
    z: -0.48547014594078064

## Orientation Implementation

start severice with frame override

```bash
ros2 run fpi_crane_rl_metrics stability_service_node \
  --ros-args -p imu_frame_override:=zed_0_left_camera_frame
```

Call service

```bash
ros2 service call \
  /get_stability \
  fpi_crane_msgs/srv/Stability \
  "{mode: 4, exponent: 4.0}"
```

Three service calls in static section of LIFT_HIGH phase (~52s)

    response:
    fpi_crane_msgs.srv.Stability_Response(success=True, message='Computed 3/4 requested stability metrics using exponent 4.0', stamp=builtin_interfaces.msg.Time(sec=1783095546, nanosec=331792000), grapple_z_base=geometry_msgs.msg.Vector3(x=-0.0008525107237444263, y=0.2058250042985906, z=0.9785884430295256), metrics=[fpi_crane_msgs.msg.StabilityMetric(reference_name='base_link_z', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=1.0), dot_product=0.9785884430295256, tilt_rad=0.2073084918746078, tilt_deg=11.877901641637147, reward=0.917065446008412, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='free_hanging_gaze', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.011373999980936902, y=-0.0367031558336826, z=0.9992614825341174), dot_product=0.9703016147082423, tilt_rad=0.2443217538706539, tilt_deg=13.998605340022554, reward=0.8863944260790548, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_acceleration', valid=False, status='No recent IMU samples passed the quasi-static acceleration and angular-rate checks', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), dot_product=0.0, tilt_rad=0.0, tilt_deg=0.0, reward=0.0, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_orientation', valid=True, status='IMU orientation estimate from 16 samples; concentration=1.0000', reference_z_base=geometry_msgs.msg.Vector3(x=-0.002101337553719874, y=0.053460358050372865, z=0.9985677615953819), dot_product=0.9881921409174242, tilt_rad=0.1538256813438785, tilt_deg=8.813562321728524, reward=0.9535985510575291, samples_used=16)])

    response:
    fpi_crane_msgs.srv.Stability_Response(success=True, message='Computed 3/4 requested stability metrics using exponent 4.0', stamp=builtin_interfaces.msg.Time(sec=1783095546, nanosec=461753000), grapple_z_base=geometry_msgs.msg.Vector3(x=-0.0059560840248185365, y=0.24011345134605072, z=0.970726560647115), metrics=[fpi_crane_msgs.msg.StabilityMetric(reference_name='base_link_z', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=1.0), dot_product=0.970726560647115, tilt_rad=0.24255880621536996, tilt_deg=13.897595879872298, reward=0.8879482407826855, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='free_hanging_gaze', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.011373999980936902, y=-0.0367031558336826, z=0.9992614825341174), dot_product=0.9611289962053787, tilt_rad=0.2797337055660343, tilt_deg=16.02756071648899, reward=0.8533490680821391, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_acceleration', valid=False, status='No recent IMU samples passed the quasi-static acceleration and angular-rate checks', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), dot_product=0.0, tilt_rad=0.0, tilt_deg=0.0, reward=0.0, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_orientation', valid=True, status='IMU orientation estimate from 40 samples; concentration=1.0000', reference_z_base=geometry_msgs.msg.Vector3(x=-0.001815240472664784, y=0.0530865226744337, z=0.9985882665104588), dot_product=0.9821137533563615, tilt_rad=0.1894192285300033, tilt_deg=10.852922353393222, reward=0.9303517341707743, samples_used=40)])

    response:
    fpi_crane_msgs.srv.Stability_Response(success=True, message='Computed 4/4 requested stability metrics using exponent 4.0', stamp=builtin_interfaces.msg.Time(sec=1783095546, nanosec=931811000), grapple_z_base=geometry_msgs.msg.Vector3(x=-0.01696548998651648, y=0.27963910006616194, z=0.9599552832625614), metrics=[fpi_crane_msgs.msg.StabilityMetric(reference_name='base_link_z', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=1.0), dot_product=0.9599552832625614, tilt_rad=0.2839537681439, tilt_deg=16.269352491481794, reward=0.8491883210269129, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='free_hanging_gaze', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.011373999980936902, y=-0.0367031558336826, z=0.9992614825341174), dot_product=0.9487897365697034, tilt_rad=0.32141379612966836, tilt_deg=18.415653995508272, reward=0.8103635763855592, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_acceleration', valid=True, status='IMU estimate from 2 samples; concentration=0.9976', reference_z_base=geometry_msgs.msg.Vector3(x=0.43365828424063296, y=0.4754591981605629, z=0.7654273599721825), dot_product=0.8603757951719553, tilt_rad=0.5347897679853891, tilt_deg=30.641196632343306, reward=0.5479648939649526, samples_used=2), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_orientation', valid=True, status='IMU orientation estimate from 14 samples; concentration=1.0000', reference_z_base=geometry_msgs.msg.Vector3(x=-0.002326532471948933, y=0.05164374331346965, z=0.9986628615419868), dot_product=0.9731527708029207, tilt_rad=0.23224222560690527, tilt_deg=13.306499352000767, reward=0.8968588419112314, samples_used=14)])

zed_imu_orientation output

    time:
    sec=1783095546, nanosec=331792000, 
    sec=1783095546, nanosec=461753000
    sec=1783095546, nanosec=931811000

    z_base_grap:
    x=-0.0008525107237444263, y=0.2058250042985906, z=0.9785884430295256
    x=-0.0059560840248185365, y=0.24011345134605072, z=0.970726560647115
    x=-0.01696548998651648, y=0.27963910006616194, z=0.9599552832625614

    z_up_reference:
    x=-0.002101337553719874, y=0.053460358050372865, z=0.9985677615953819
    x=-0.001815240472664784, y=0.0530865226744337, z=0.9985882665104588
    x=-0.002326532471948933, y=0.05164374331346965, z=0.9986628615419868

    reward:
    tilt_deg=8.813562321728524, reward=0.9535985510575291
    tilt_deg=10.852922353393222, reward=0.9303517341707743
    tilt_deg=13.306499352000767, reward=0.8968588419112314

## 07/13/2026 - Consistent Measurement

extract data from rosbag
```bash
python3 extract_lift_high_data.py \
  /media/fpiadmin/Lucas_Drive/crane_run_20260703_161820 \
  --out lift_high_extract_0_joint \
  --interval-index 0
```


    response:
    fpi_crane_msgs.srv.Stability_Response(success=True, message='Computed 3/4 requested stability metrics using exponent 4.0', stamp=builtin_interfaces.msg.Time(sec=1783095547, nanosec=781817000), grapple_z_base=geometry_msgs.msg.Vector3(x=0.003907139166459003, y=0.14220882520954425, z=0.9898289671938558), metrics=[fpi_crane_msgs.msg.StabilityMetric(reference_name='base_link_z', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=1.0), dot_product=0.9898289671938558, tilt_rad=0.14274677842305963, tilt_deg=8.178787942730441, reward=0.9599323701577226, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='free_hanging_gaze', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.011373999980936902, y=-0.0367031558336826, z=0.9992614825341174), dot_product=0.9839228883415606, tilt_rad=0.17955710082001314, tilt_deg=10.287864058591765, reward=0.9372258392599494, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_acceleration', valid=False, status='No recent IMU samples passed the quasi-static acceleration and angular-rate checks', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), dot_product=0.0, tilt_rad=0.0, tilt_deg=0.0, reward=0.0, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_orientation', valid=True, status='IMU orientation estimate from 15 samples; concentration=1.0000', reference_z_base=geometry_msgs.msg.Vector3(x=-0.0026641005440894627, y=0.05237012723300064, z=0.998624189744015), dot_product=0.9959042356074197, tilt_rad=0.09053798463928296, tilt_deg=5.187444405451191, reward=0.9837173195963705, samples_used=15)])


    response:
    fpi_crane_msgs.srv.Stability_Response(success=True, message='Computed 3/4 requested stability metrics using exponent 4.0', stamp=builtin_interfaces.msg.Time(sec=1783095550, nanosec=596800000), grapple_z_base=geometry_msgs.msg.Vector3(x=0.4896583437945775, y=-0.0863183861086117, z=0.8676311673585476), metrics=[fpi_crane_msgs.msg.StabilityMetric(reference_name='base_link_z', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=1.0), dot_product=0.8676311673585476, tilt_rad=0.5203782607212023, tilt_deg=29.815478089683275, reward=0.5666835136373995, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='free_hanging_gaze', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.011373999980936902, y=-0.0367031558336826, z=0.9992614825341174), dot_product=0.8757279377571507, tilt_rad=0.5038548392299802, tilt_deg=28.868755775120487, reward=0.588134721881655, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_acceleration', valid=False, status='No recent IMU samples passed the quasi-static acceleration and angular-rate checks', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), dot_product=0.0, tilt_rad=0.0, tilt_deg=0.0, reward=0.0, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_orientation', valid=True, status='IMU orientation estimate from 4 samples; concentration=1.0000', reference_z_base=geometry_msgs.msg.Vector3(x=0.0016358985095081856, y=0.054500990754894875, z=0.9985123764094272), dot_product=0.8624370525575679, tilt_rad=0.5307314792763019, tilt_deg=30.408673817267015, reward=0.5532349733383684, samples_used=4)])

    response:
    fpi_crane_msgs.srv.Stability_Response(success=True, message='Computed 3/4 requested stability metrics using exponent 4.0', stamp=builtin_interfaces.msg.Time(sec=1783095552, nanosec=661776000), grapple_z_base=geometry_msgs.msg.Vector3(x=0.429784295565263, y=-0.08421562240010824, z=0.8989956552894085), metrics=[fpi_crane_msgs.msg.StabilityMetric(reference_name='base_link_z', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=1.0), dot_product=0.8989956552894085, tilt_rad=0.45332548367513614, tilt_deg=25.973636960312, reward=0.6531762295001324, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='free_hanging_gaze', valid=True, status='OK', reference_z_base=geometry_msgs.msg.Vector3(x=0.011373999980936902, y=-0.0367031558336826, z=0.9992614825341174), dot_product=0.9063110769783727, tilt_rad=0.4363245282682889, tilt_deg=24.999553967809536, reward=0.6746975792872283, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_acceleration', valid=False, status='No recent IMU samples passed the quasi-static acceleration and angular-rate checks', reference_z_base=geometry_msgs.msg.Vector3(x=0.0, y=0.0, z=0.0), dot_product=0.0, tilt_rad=0.0, tilt_deg=0.0, reward=0.0, samples_used=0), fpi_crane_msgs.msg.StabilityMetric(reference_name='zed_imu_orientation', valid=True, status='IMU orientation estimate from 8 samples; concentration=1.0000', reference_z_base=geometry_msgs.msg.Vector3(x=0.00026672240363122496, y=0.05777305115588246, z=0.9983297067699124), dot_product=0.8927433085718106, tilt_rad=0.4673988217928431, tilt_deg=26.779979838117196, reward=0.6351940403976687, samples_used=8)])

Summary:

    time:
    sec=1783095547, nanosec=781817000, elasped:46.619995257s
    sec=1783095550, nanosec=596800000, elasped:49.430257064
    sec=1783095552, nanosec=661776000, elasped: 51.500034326


    z_base_grap:
    x=0.003907139166459003, y=0.14220882520954425, z=0.9898289671938558
    x=0.4896583437945775, y=-0.0863183861086117, z=0.8676311673585476
    x=0.429784295565263, y=-0.08421562240010824, z=0.8989956552894085

Service uses historical JointState data that does not align with the static portion of the LIFT_HIGH phase. 

Mean ẑ base_link to grapplecarrier from FK on joint state topic during static period:

    Z-axis means - X: 0.43184146809505114, Y: -0.06953385368484172, Z: 0.8988766136524025

Mean ẑ up in IMU frame from IMU orientation topic during static period:

    Estimated gravity direction in IMU frame: [-0.33473503  0.00869411  0.94227219]


Transform form camera_left to base_link during static period from `ros2 run tf2_ros tf2_echo base_link zed_0_left_camera_frame`

    At time 1783095553.265456000
    - Translation: [-0.077, -0.218, 1.361]
    - Rotation: in Quaternion (xyzw) [-0.164, -0.002, 0.968, -0.189]
    - Rotation: in RPY (radian) [0.062, 0.325, -2.746]
    - Rotation: in RPY (degree) [3.567, 18.593, -157.345]
    - Matrix:
    -0.875  0.366 -0.318 -0.077
    -0.365 -0.929 -0.065 -0.218
    -0.319  0.059  0.946  1.361
    0.000  0.000  0.000  1.000
    At time 1783095554.265422000
    - Translation: [-0.076, -0.218, 1.361]
    - Rotation: in Quaternion (xyzw) [-0.164, -0.002, 0.968, -0.191]
    - Rotation: in RPY (radian) [0.062, 0.325, -2.743]
    - Rotation: in RPY (degree) [3.567, 18.593, -157.145]
    - Matrix:
    -0.873  0.369 -0.317 -0.076
    -0.368 -0.927 -0.066 -0.218
    -0.319  0.059  0.946  1.361
    0.000  0.000  0.000  1.000

Constant transform from imu to camera left on extra ZED21 from `ros2 run tf2_ros tf2_echo zed_left_camera_frame zed_imu_link`

    At time 1783536822.138282290
    - Translation: [-0.002, -0.023, 0.000]
    - Rotation: in Quaternion (xyzw) [0.001, 0.000, -0.005, 1.000]
    - Rotation: in RPY (radian) [0.002, 0.000, -0.011]
    - Rotation: in RPY (degree) [0.137, 0.026, -0.618]
    - Matrix:
    1.000  0.011  0.000 -0.002
    -0.011  1.000 -0.002 -0.023
    -0.000  0.002  1.000  0.000
    0.000  0.000  0.000  1.000

## 07/15/2026 FSM Implementation

`MeasureStability` Action server is called from fsm with the following sequence: 

`LIFT_HIGH` reached by existing authoritative `_check_traj_done()` -> `SEQ_WAIT_STABILITY` -> send one `MeasureStability` goal -> hold `LIFT_HIGH` by sending no new trajectory -> action success -> `_advance_move()` -> existing logic sends `CARRY_HOME`.

Start action server node

```bash
ros2 run fpi_crane_rl_metrics measure_stability_action_node --ros-args -p use_sim_time:=true -p imu_frame_override:=zed_0_left_camera_frame
```
