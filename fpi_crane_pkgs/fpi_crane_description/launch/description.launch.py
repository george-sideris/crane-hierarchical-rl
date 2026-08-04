import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.parameter_descriptions import ParameterValue





def generate_launch_description():
    viz = LaunchConfiguration('viz')
    sim = LaunchConfiguration('sim')

    package_name = 'fpi_crane_description'
    xacro_file_name = 'fpi_crane.urdf.xacro'

    # Get the share directory of the package containing your URDF
    pkg_dir = get_package_share_directory(package_name)
    robot_xacro = os.path.join(pkg_dir, 'urdf', xacro_file_name)

    robot_description = ParameterValue(
        Command([
            "xacro ",
            robot_xacro,
        ]),
        value_type=str,
    )

    # Node for the robot_state_publisher in hardware mode.
    robot_state_publisher_hw = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description},
                    {'use_sim_time': False}],
        condition=UnlessCondition(sim),
        output='screen'
    )

    # Node for the robot_state_publisher in simulation mode.
    robot_state_publisher_sim = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{'robot_description': robot_description},
                    {'use_sim_time': False}],
        remappings=[('joint_states', 'joint_states_sim')],
        condition=IfCondition(sim),
        output='screen'
    )

    # Node for RViz2
    rviz2_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(pkg_dir, 'rviz', 'config.rviz')],
        condition=IfCondition(viz),
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
            description='Use simulation joint state remapping.',
        ),

        robot_state_publisher_hw,
        robot_state_publisher_sim,
        rviz2_node
    ])
