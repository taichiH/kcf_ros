cmake_minimum_required(VERSION 2.8)

project(kcf_ros)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic -O3 -Wno-long-long -fno-omit-frame-pointer")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  image_transport
  sensor_msgs
)

find_package(OpenCV REQUIRED)
link_directories (${OpenCV_LIB_DIR})
message(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

include_directories (
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  )

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

include_directories(${CMAKE_BINARY_DIR}/)
include_directories(
  ${CMAKE_SOURCE_DIR}/libs/src/
  ${CMAKE_SOURCE_DIR}/libs/src/piotr_fhog
  ${CMAKE_SOURCE_DIR}/libs/src/cn
  )

catkin_package(
  CATKIN_DEPENDS
  roscpp
  rospy
  cv_bridge
  image_transport
  sensor_msgs
  )

add_subdirectory(libs/src)

add_executable(kcf_tracker src/kcf_tracker.cpp src/vot.hpp)
target_link_libraries(kcf_tracker ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} kcf)
