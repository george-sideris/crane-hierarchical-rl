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
    joy = LaunchConfiguration('joy')

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
        }.items(),
    )

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        condition=IfCondition(joy),
        output='screen',
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
            'joy',
            default_value='false',
            description='Launch the ROS 2 joy_node.',
        ),

        robot_description_launch,
        plc_launch,
        joy_node,
    ])
