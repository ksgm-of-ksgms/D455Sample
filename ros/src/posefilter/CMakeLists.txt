cmake_minimum_required(VERSION 3.0.2)
project(posefilter)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
)
set(Ceres_DIR "/usr/local/lib/cmake/Ceres/")
find_package(Ceres REQUIRED)
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
include_directories("/usr/include/eigen3")
include_directories("/usr/include/pcl-1.7")
include_directories("/usr/local/include/ceres")

add_executable(posefilter src/posefilter.cpp)

target_link_libraries(
  posefilter
  ${catkin_LIBRARIES}
  ceres
)
