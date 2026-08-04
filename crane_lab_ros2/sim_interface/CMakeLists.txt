cmake_minimum_required(VERSION 3.8)
project(sim_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

#### ROS2 Interface Directives ####
set(interface_files
  "srv/SetTarget.srv"
  "srv/CheckTargetReached.srv"
  "action/FollowPath.action"
  "msg/Waypoint.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${interface_files}
  DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(
  rosidl_default_runtime
)

install(
  PROGRAMS
    sim_interface/sim_velocity_control_ui_node.py
    sim_interface/joint_velocity_controller_node.py
    sim_interface/trajectory_execution_action_server.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

install(
  DIRECTORY sim_interface/
  DESTINATION local/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/dist-packages/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
