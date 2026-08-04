"""Replay a rosbag against the calibrated URDF in RViz to validate camera extrinsics.

Runs robot_state_publisher (replay:=true so it emits the zed_0 camera-internal frames)
+ RViz on sim time, and plays the bag with its /tf and /tf_static remapped away so
robot_state_publisher is the SOLE tf authority (no stale recorded extrinsics fighting it).

Usage:
  ros2 launch fpi_crane_description replay_validate.launch.py \
      bag:=/workspace/crane_testbed/rosbag2_cranelab_test_2026_06_26-16_48_33
  add loop:=true to repeat (there is a brief tf reset each loop).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_REMAP = ['--remap', '/tf:=/tf_old', '/tf_static:=/tf_static_old']


def generate_launch_description():
    bag = LaunchConfiguration('bag')
    loop = LaunchConfiguration('loop')

    description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('fpi_crane_description'), 'launch', 'description.launch.py'])),
        launch_arguments={'viz': 'true', 'use_sim_time': 'true', 'replay': 'true'}.items(),
    )

    play_once = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag, '--clock'] + _REMAP,
        output='screen', condition=UnlessCondition(loop),
    )
    play_loop = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', bag, '--clock', '-l'] + _REMAP,
        output='screen', condition=IfCondition(loop),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'bag',
            default_value='/workspace/crane_testbed/rosbag2_cranelab_test_2026_06_26-16_48_33',
            description='Path to the rosbag2 directory to replay.',
        ),
        DeclareLaunchArgument(
            'loop', default_value='false',
            description='Loop the bag (brief tf reset each cycle) vs play once paused.',
        ),
        description,
        play_once,
        play_loop,
    ])
