from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_simulation = LaunchConfiguration("use_simulation")
    udp_recv_ip_address = LaunchConfiguration("udp_recv_ip_address")
    udp_recv_port = LaunchConfiguration("udp_recv_port")
    udp_send_ip_address = LaunchConfiguration("udp_send_ip_address")
    udp_send_port = LaunchConfiguration("udp_send_port")
    telescope_static = LaunchConfiguration("telescope_static")
    telescope_static_value = LaunchConfiguration("telescope_static_value")

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_simulation", default_value="False"),
            DeclareLaunchArgument("udp_recv_ip_address", default_value="172.20.230.162"),
            DeclareLaunchArgument("udp_recv_port", default_value="30305"),
            DeclareLaunchArgument("udp_send_ip_address", default_value="172.20.230.120"),
            DeclareLaunchArgument("udp_send_port", default_value="30310"),
            DeclareLaunchArgument(
                "telescope_static",
                default_value="False",
                description="Hold the telescope joint static (use when its encoder is "
                            "unreliable). Must match crane_policy_node's telescope_static.",
            ),
            DeclareLaunchArgument(
                "telescope_static_value",
                default_value="0.13",
                description="Position (m) to hold the telescope at when telescope_static is true.",
            ),
            Node(
                package="fpi_crane_hw",
                executable="plc_node",
                name="plc_node",
                output="screen",
                parameters=[
                    {
                        "use_simulation": ParameterValue(use_simulation, value_type=bool),
                        "udp_recv_ip_address": udp_recv_ip_address,
                        "udp_recv_port": ParameterValue(udp_recv_port, value_type=int),
                        "udp_send_ip_address": udp_send_ip_address,
                        "udp_send_port": ParameterValue(udp_send_port, value_type=int),
                        "telescope_static": ParameterValue(telescope_static, value_type=bool),
                        "telescope_static_value": ParameterValue(
                            telescope_static_value, value_type=float),
                    }
                ],
            ),
        ]
    )
