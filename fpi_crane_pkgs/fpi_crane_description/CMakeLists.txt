cmake_minimum_required(VERSION 3.10)
project(fpi_crane_description)
find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf launch rviz DESTINATION share/${PROJECT_NAME})

if(AMENT_CMAKE_SYMLINK_INSTALL)
  install(CODE
    "file(MAKE_DIRECTORY \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}\")")
  install(CODE
    "file(REMOVE_RECURSE \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/meshes\")")
  install(CODE
    "execute_process(COMMAND \"${CMAKE_COMMAND}\" -E create_symlink
      \"${CMAKE_CURRENT_SOURCE_DIR}/meshes\"
      \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/meshes\")")
else()
  install(DIRECTORY meshes DESTINATION share/${PROJECT_NAME})
endif()

ament_package()