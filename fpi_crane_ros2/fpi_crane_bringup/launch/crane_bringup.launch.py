import os
import shutil

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


ROBOT_DESCRIPTION_PACKAGE = 'fpi_crane_description'
ROBOT_DESCRIPTION_LAUNCH_FILE = 'description.launch.py'

PLC_PACKAGE = 'fpi_crane_hw'
PLC_LAUNCH_FILE = 'plc_node.launch.py'


def generate_launch_description():
    viz = LaunchConfiguration('viz')
    sim = LaunchConfiguration('sim')
    use_sim_time = LaunchConfiguration('use_sim_time')
    joy = LaunchConfiguration('joy')
    keyboard = LaunchConfiguration('keyboard')
    telescope_static = LaunchConfiguration('telescope_static')
    telescope_static_value = LaunchConfiguration('telescope_static_value')

    robot_description_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare(ROBOT_DESCRIPTION_PACKAGE),
                'launch',
                ROBOT_DESCRIPTION_LAUNCH_FILE,
            ])
        ),
        launch_arguments={
            'viz': viz,
            'sim': sim,
            'use_sim_time': use_sim_time,
        }.items(),
    )

    plc_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare(PLC_PACKAGE),
                'launch',
                PLC_LAUNCH_FILE,
            ])
        ),
        launch_arguments={
            'use_simulation': sim,
            'use_sim_time': use_sim_time,
            'telescope_static': telescope_static,
            'telescope_static_value': telescope_static_value,
        }.items(),
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        condition=IfCondition(joy),
        output='screen',
    )

    keyboard_prefix = []
    if shutil.which('gnome-terminal') and (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        keyboard_prefix = ['gnome-terminal', '--']

    keyboard_joy_node = Node(
        package='fpi_crane_hw',
        executable='keyboard_joy_node',
        name='keyboard_joy_node',
        condition=IfCondition(keyboard),
        output='screen',
        prefix=keyboard_prefix,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'viz',
            default_value='true',
            description='Launch RViz with the robot_bringup RViz configuration.',
        ),
        DeclareLaunchArgument(
            'sim',
            default_value='false',
            description='Launch the PLC launch file in simulation mode.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use the /clock topic for robot_state_publisher and RViz.',
        ),
        DeclareLaunchArgument(
            'joy',
            default_value='false',
            description='Launch the ROS 2 joy_node.',
        ),
        DeclareLaunchArgument(
            'keyboard',
            default_value='false',
            description='Publish a keyboard-emulated /joy topic instead of using a physical controller.',
        ),
        DeclareLaunchArgument(
            'telescope_static',
            default_value='false',
            description='Hold the telescope joint static in the PLC node (use when its '
                        'encoder is unreliable). NOTE: crane_policy_node is launched '
                        'separately and must be given a matching -p telescope_static:=true.',
        ),
        DeclareLaunchArgument(
            'telescope_static_value',
            default_value='0.13',
            description='Position (m) to hold the telescope at when telescope_static is true.',
        ),

        robot_description_launch,
        plc_launch,
        joy_node,
        keyboard_joy_node,
    ])
