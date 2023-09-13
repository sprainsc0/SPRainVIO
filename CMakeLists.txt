cmake_minimum_required(VERSION 3.5)
project(msckf_node)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_BUILD_TYPE Release)

include(EnableC++XX)

# add_definitions(-DUSE_CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# Check if any sanitizers set
include(EnableSanitizers)
# include(FindSuiteSparse)

# Check if ROS_DISTRO is greater or equal to Dashing
# Required since
#   - "create_subscription()" and "create_publisher()" APIs changed
#   - eigen3_cmake_module is only available in these distros
set(ROS_DISTRO $ENV{ROS2_DISTRO})
set(PCL_INCLUDE_DIR /usr/include/pcl-1.8)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rclcpp REQUIRED)

find_package(nav_msgs REQUIRED)
find_package(sprain_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(std_srvs REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(SuiteSparse REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(PCL REQUIRED COMPONENTS)
find_package(pcl_conversions REQUIRED)

find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV 4 REQUIRED)

#################
# Setup targets #
#################

include_directories(
  include
  ${Boost_INCLUDE_DIR}
  ${SUITESPARSE_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Add comm_library lib
add_library(comm_library SHARED src/common/math_library.cpp
      src/common/quaternion.cpp
      src/common/signal_link.cpp
      src/common/random_numbers.cpp
      src/common/perform.cpp)
ament_target_dependencies(comm_library Eigen3 rclcpp)
target_include_directories(comm_library PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/common>
  $<INSTALL_INTERFACE:include/common>
  ${Eigen3_INCLUDE_DIRS}
)
target_link_libraries(comm_library Eigen3::Eigen)

add_library(image_track SHARED src/image_track/image_track.cpp)
ament_target_dependencies(image_track message_filters image_transport cv_bridge std_msgs sensor_msgs sprain_msgs rclcpp Eigen3)
target_include_directories(image_track PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/image_track>
  $<INSTALL_INTERFACE:include/image_track>
  ${Eigen3_INCLUDE_DIRS}
)
target_link_libraries(image_track comm_library Eigen3::Eigen ${OpenCV_LIBRARIES})

add_library(msckf_core SHARED 
  src/ekf_core/ekf_Fusion.cpp
  src/ekf_core/ekf_FuseExtAtt.cpp
  src/ekf_core/ekf_FuseNEDPos.cpp
  src/ekf_core/ekf_FuseNEDVel.cpp
  src/ekf_core/ekf_FuseFeature.cpp
  src/ekf_core/ekf_Core.cpp
  src/ekf_core/ekf_LogicConst.cpp
  src/ekf_core/ekf_LogicAtt.cpp
  src/ekf_core/ekf_LogicCam.cpp
  src/ekf_core/ekf_Param.cpp
  src/ekf_core/ekf_StateOut.cpp)
ament_target_dependencies(msckf_core Eigen3 Boost message_filters image_transport cv_bridge rclcpp nav_msgs geometry_msgs sprain_msgs tf2_ros pcl_conversions SUITESPARSE)
target_include_directories(msckf_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/msckf_core>
  $<INSTALL_INTERFACE:include/msckf_core>
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(msckf_core Eigen3::Eigen image_track comm_library ${Boost_SYSTEM_LIBRARY} ${OpenCV_LIBRARIES})

add_executable(msckf_node src/msckf_node.cpp)
ament_target_dependencies(msckf_node sprain_msgs rclcpp)
target_include_directories(msckf_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(msckf_node Eigen3::Eigen comm_library msckf_core)

ament_export_dependencies(ament_cmake rclcpp rosidl_default_runtime cv_bridge eigen3_cmake_module Eigen3 sprain_msgs geometry_msgs sensor_msgs)

ament_export_targets(export_msckf_core HAS_LIBRARY_TARGET)
ament_export_targets(export_comm_library HAS_LIBRARY_TARGET)
ament_export_targets(export_image_track HAS_LIBRARY_TARGET)

install(DIRECTORY cmake
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS comm_library
  EXPORT export_comm_library
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS image_track
  EXPORT export_image_track
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS msckf_core
  EXPORT export_msckf_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS msckf_node
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
