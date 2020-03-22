ros_tensorflow tutorial
=====

This repo implements a Tensorflow 2 integration in ROS.
Explanations are given in the following [blog post](https://jacqueskaiser.com/posts/2020/03/ros-tensorflow).

Files
-----

This repo contains two ROS packages:
- `ros_tensorflow`
- `ros_tensorflow_msgs`

Installing
-----

1. Install the dependencies: ```pip3 install -r requirements.txt```
2. Clone this repo to your catkin workspace
3. Build the workspace: `catkin_make`

Running
-----

1. Run the node: `rosrun ros_tensorflow node.py`
Predictions are performed at regular time interval in a loop.

2. Train the Tensorflow model (use tab completion):
```bash
rostopic pub /train/goal ros_tensorflow_msgs/TrainActionGoal "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
goal_id:
  stamp:
    secs: 0
    nsecs: 0
  id: ''
goal:
  epochs: 10"
```

3. Abort training (use tab completion):
```bash
rostopic pub /train/cancel actionlib_msgs/GoalID "stamp:
  secs: 0
  nsecs: 0
id: ''"
```

4. Run predicitons from a service call (use tab completion):
```bash
rosservice call /predict "data: [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]"
```
