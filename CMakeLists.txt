cmake_minimum_required(VERSION 2.8)
project(kcf_ros)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer -fpic")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  nodelet
  message_generation
  eigen_conversions
  autoware_msgs
)

find_package(OpenCV REQUIRED)
link_directories (${OpenCV_LIB_DIR})
message(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")

include_directories (
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/
  ${CMAKE_SOURCE_DIR}/libs/src/
  ${CMAKE_SOURCE_DIR}/libs/src/piotr_fhog
  ${CMAKE_SOURCE_DIR}/libs/src/cn
  )

add_message_files(
  FILES
  Rect.msg
  )

generate_messages(
  DEPENDENCIES
  std_msgs
  )

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  autoware_msgs
  )

add_subdirectory(libs/src)

# add_executable(kcf_tracker src/kcf_tracker.cpp src/vot.hpp)
add_library(${PROJECT_NAME} SHARED
  src/kcf_tracker.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${EIGEN_LIBRARIES} kcf)
