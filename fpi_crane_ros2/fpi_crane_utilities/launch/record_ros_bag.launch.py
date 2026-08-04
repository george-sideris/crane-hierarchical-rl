from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
import datetime


def generate_launch_description():


    location_arg = LaunchConfiguration('location')
    test_arg = LaunchConfiguration('test')

    # Generate timestamp once at launch start
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    bag_name = PythonExpression([
    f'"rosbag2_" + "',
    location_arg,
    '" + "_" + "',
    test_arg,
    '" + "_" + "',
    timestamp,
    '"'
    ])

    topics = [
        
        # general
        '/tf',
        '/tf_static',
        '/joint_states',
        '/plc_status',
        '/RobotTrajectoryInfo',
        
        # stick camera
        '/zed_1/zed_node/rgb/color/rect/camera_info',
        '/zed_1/zed_node/rgb/color/rect/image',
        '/zed_1/zed_node/point_cloud/cloud_registered',
        
        # mast camera
        '/zed_0/zed_node/rgb/color/rect/camera_info',
        '/zed_0/zed_node/rgb/color/rect/image',
        '/zed_0/zed_node/point_cloud/cloud_registered',
        
        # mast lidar
        '/lidar_0/points'
    ]
        

    return LaunchDescription([

        DeclareLaunchArgument(
            'location',
            default_value='cranelab',
            description='Location to describe in bag file name'
        ),

        DeclareLaunchArgument(
            'test',
            default_value='calibration',
            description='Test name to describe in bag file name'
        ),

        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '-o', bag_name
            ] + topics,
            output='screen'
        )
    ])