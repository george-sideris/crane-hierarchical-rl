cmake_minimum_required(VERSION 3.8)
project(fpi_crane_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PlcStatus.msg"
  "msg/PlcGripper.msg"
  "msg/PlcStop.msg"
  "msg/RobotTrajectoryInfo.msg"
  "msg/StabilityMetric.msg"
  "action/MeasureStability.action"
  "srv/RequestGrappleMove.srv"
  "srv/RelativeJointMove.srv"
  "srv/GrappleFK.srv"
  "srv/Stability.srv"
  DEPENDENCIES
    std_msgs
    sensor_msgs
    trajectory_msgs
    builtin_interfaces
    geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
