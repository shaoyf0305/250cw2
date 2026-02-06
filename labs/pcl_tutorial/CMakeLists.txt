cmake_minimum_required(VERSION 3.8)
project(pcl_tutorial)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation features search)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)

add_executable(pcl_tutorial_node src/pcl_tutorial_node.cpp)
ament_target_dependencies(
  pcl_tutorial_node
  rclcpp
  sensor_msgs
  geometry_msgs
  pcl_conversions
  pcl_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)
target_link_libraries(pcl_tutorial_node ${PCL_LIBRARIES})
target_compile_definitions(pcl_tutorial_node PRIVATE ${PCL_DEFINITIONS})

install(TARGETS pcl_tutorial_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
