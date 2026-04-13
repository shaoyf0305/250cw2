# COMP0250 Coursework 1 - Team 36

## Authors & Contribution

| Student Name | Task 1 | Task 2 | Task 3 | Total Hours | Percentage |

| [YIXIAO TAO] | 3 hrs  | 4 hrs | X hrs | 7 hrs | 33.3% |

| [YIFAN SHAO] | 1 hrs  | 0 hrs | 6 hrs | 7 hrs | 33.3% |

| [ QI CHENG ] | 3 hrs  | 1 hrs | 3 hrs | 7 hrs | 33.3% |


## License
This project is licensed under the MIT License - see the `package.xml` file for details.

## Package Description
This package (`cw1_team_36`) contains the ROS 2 C++ solution for Coursework 1 of COMP0250. It controls a Panda robotic arm in a Gazebo simulation to perform pick-and-place tasks, color identification, and object sorting using point cloud data from an RGB-D camera.

### Key Features & Design Choices:
- **Collision Avoidance**: The robot uses MoveIt for collision-free trajectory planning to prevent collisions with the ground or baskets.
- **Robust Color Detection (HSV)**: Instead of raw RGB, the vision pipeline converts point cloud colors to HSV space. This makes color classification (Red, Blue, Purple) highly robust against shadows and reflections.
- **Multi-view Fusion**: the robot scans the workspace from 6 different poses. Detections are merged across frames using spatial distance thresholds
- **Majority Voting**: The final color of an object is determined by a majority vote of its highest points across all observation frames, further improving robustness.
- **Shape Analysis**: Baskets and cubes are distinguished by analyzing the geometric shape of the point cloud clusters (e.g., checking for a hollow center and a raised rim).






## How to Build

1. Navigate to the root of your workspace (e.g., `~/comp0250_s26_labs`).
2. Build the package using `colcon`:

```bash
cd ~/comp0250_s26_labs
colcon build --packages-select cw1_team_36
```

3. Source the newly built workspace:

```bash
cd ~/comp0250_S26_labs
source /opt/ros/humble/setup.bash
source install/setup.bash
export PATH=/usr/bin:$PATH
export RMW_FASTRTPS_USE_SHM=0
```

## How to Run

1. **Launch the Simulation Environment**:
   The recommended command to avoid possible Rviz errors and additional information in terminal is:

```bash
stdbuf -oL ros2 launch cw1_team_36 run_solution.launch.py \
  use_gazebo_gui:=true use_rviz:=false \
  enable_realsense:=true enable_camera_processing:=false \
  control_mode:=effort \
| grep --line-buffered '\[cw1_solution_node\]:' \
| sed -u 's/^.*\[cw1_solution_node\]: //'
```

2. **Trigger the Tasks**:
   In a third terminal, use the test launch file to trigger the specific task you want to evaluate (Task 1, 2, or 3):

```bash
# For Task 1
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 1}"

# For Task 2
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 2}"

# For Task 3
ros2 service call /task cw1_world_spawner/srv/TaskSetup "{task_index: 3}"
```

3. **Additional information**:

   Use Rviz seperately:

```bash
# Replace the file default.rviz in ~/.rviz2 with the provided version to check camera point cloud
rviz2
```

   If gazebo is stuck:

```bash
pkill -f rviz
pkill -f gazebo
pkill -f gzserver
pkill -f gzclient
```