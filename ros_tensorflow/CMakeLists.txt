cmake_minimum_required(VERSION 2.8.3)
project(ros_tensorflow)

find_package(catkin REQUIRED COMPONENTS
  ros_tensorflow_msgs
  actionlib
  rospy
  std_msgs
)

catkin_python_setup()

catkin_package()
include_directories(${catkin_INCLUDE_DIRS})
