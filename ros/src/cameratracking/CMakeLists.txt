cmake_minimum_required(VERSION 3.0.2)
project(cameratracking)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  cv_bridge
  image_transport
  sensor_msgs
)

find_package(OpenCV REQUIRED)

set(Ceres_DIR "/usr/local/lib/cmake/Ceres/")

find_package(Ceres REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
include_directories("include")
include_directories("/usr/include/eigen3")
include_directories("/usr/include/pcl-1.7")
include_directories("/usr/local/include/ceres")

add_executable(${PROJECT_NAME} src/${PROJECT_NAME}.cpp)
target_link_libraries(
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ceres
)

catkin_add_gtest(${PROJECT_NAME}_test test/utest.cpp)

target_link_libraries(${PROJECT_NAME}_test
                      ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ceres
)

