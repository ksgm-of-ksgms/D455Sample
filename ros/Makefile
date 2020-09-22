#http://wiki.ros.org/ja/catkin/Tutorials/create_a_workspace

build-release:
	catkin build -DCMAKE_BUILD_TYPE=Release

build-debug:
	catkin build -DCMAKE_BUILD_TYPE=Debug

clean:
	catkin clean

init:
	@echo source devel/setup.bash

rec:
	make -C src/posefilter run
run:
	#make -C src/posefilter run
	make -C src/cameratracking run
	#roslaunch realsense2_camera rs_camera.launch enable_accel:="true" align_depth:="true" enable_gyro:="true" unite_imu_method:="copy"

test:
	catkin run_tests
	#gdb ./devel/lib/cameratracking/cameratracking_test
	./devel/lib/cameratracking/cameratracking_test


pub:
	rostopic pub -r 0.1  /acccalib  geometry_msgs/Vector3 -- '{x: 0, y: 0, z: -0.5}'

rviz:
	rosrun rviz rviz

config:
	rosrun rqt_reconfigure rqt_reconfigure

.PHONY:build
