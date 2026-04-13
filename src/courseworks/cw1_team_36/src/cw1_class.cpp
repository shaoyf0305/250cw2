/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire 
solution is contained within the cw1_team_<your_team_number> package */

#include <cw1_class.h>
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include <cmath>
#include <limits>
#include <queue>
#include <string>
#include <vector>

#include <moveit/planning_interface/planning_interface.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <rmw/qos_profiles.h>
///////////////////////////////////////////////////////////////////////////////

cw1::cw1(const rclcpp::Node::SharedPtr &node)
{
  /* class constructor */
  // Initialize ROS node and create mutually exclusive callback groups
  // to prevent service and sensor callbacks from blocking each other.
  node_ = node;
  service_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  sensor_cb_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // advertise solutions for coursework tasks
  t1_service_ = node_->create_service<cw1_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw1::t1_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);
  t2_service_ = node_->create_service<cw1_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw1::t2_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);
  t3_service_ = node_->create_service<cw1_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw1::t3_callback, this, std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, service_cb_group_);

  // Service and sensor callbacks use separate callback groups to align with the
  // current runtime architecture used in cw1_team_0.
  rclcpp::SubscriptionOptions joint_state_sub_options;
  joint_state_sub_options.callback_group = sensor_cb_group_;
  auto joint_state_qos = rclcpp::QoS(rclcpp::KeepLast(50));
  joint_state_qos.reliable();
  joint_state_qos.durability_volatile();
  joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", joint_state_qos,
    [this](const sensor_msgs::msg::JointState::ConstSharedPtr msg) {
      const int64_t stamp_ns =
        static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
      latest_joint_state_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
      joint_state_msg_count_.fetch_add(1, std::memory_order_relaxed);
    },
    joint_state_sub_options);

  rclcpp::SubscriptionOptions cloud_sub_options;
  cloud_sub_options.callback_group = sensor_cb_group_;
  auto cloud_qos = rclcpp::QoS(rclcpp::KeepLast(10));
  cloud_qos.reliable();
  cloud_qos.durability_volatile();
  cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/r200/camera/depth_registered/points", cloud_qos,
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        latest_cloud_ = msg;
      }
      const int64_t stamp_ns =
        static_cast<int64_t>(msg->header.stamp.sec) * 1000000000LL +
        static_cast<int64_t>(msg->header.stamp.nanosec);
      latest_cloud_stamp_ns_.store(stamp_ns, std::memory_order_relaxed);
      cloud_msg_count_.fetch_add(1, std::memory_order_relaxed);
    },
    cloud_sub_options);

  // Parameter declarations intentionally mirror cw1_team_0 for compatibility.
  const bool use_gazebo_gui = node_->declare_parameter<bool>("use_gazebo_gui", true);
  (void)use_gazebo_gui;
  enable_cloud_viewer_ = node_->declare_parameter<bool>("enable_cloud_viewer", false);
  move_home_on_start_ = node_->declare_parameter<bool>("move_home_on_start", false);
  use_path_constraints_ = node_->declare_parameter<bool>("use_path_constraints", false);
  use_cartesian_reach_ = node_->declare_parameter<bool>("use_cartesian_reach", false);
  allow_position_only_fallback_ = node_->declare_parameter<bool>(
    "allow_position_only_fallback", allow_position_only_fallback_);
  cartesian_eef_step_ = node_->declare_parameter<double>(
    "cartesian_eef_step", cartesian_eef_step_);
  cartesian_jump_threshold_ = node_->declare_parameter<double>(
    "cartesian_jump_threshold", cartesian_jump_threshold_);
  cartesian_min_fraction_ = node_->declare_parameter<double>(
    "cartesian_min_fraction", cartesian_min_fraction_);
  publish_programmatic_debug_ = node_->declare_parameter<bool>(
    "publish_programmatic_debug", publish_programmatic_debug_);
  enable_task1_snap_ = node_->declare_parameter<bool>("enable_task1_snap", false);
  return_home_between_pick_place_ = node_->declare_parameter<bool>(
    "return_home_between_pick_place", return_home_between_pick_place_);
  return_home_after_pick_place_ = node_->declare_parameter<bool>(
    "return_home_after_pick_place", return_home_after_pick_place_);
  pick_offset_z_ = node_->declare_parameter<double>("pick_offset_z", pick_offset_z_);
  task3_pick_offset_z_ = node_->declare_parameter<double>(
    "task3_pick_offset_z", task3_pick_offset_z_);
  task2_capture_enabled_ = node_->declare_parameter<bool>(
    "task2_capture_enabled", task2_capture_enabled_);
  task2_capture_dir_ = node_->declare_parameter<std::string>(
    "task2_capture_dir", task2_capture_dir_);
  place_offset_z_ = node_->declare_parameter<double>("place_offset_z", place_offset_z_);
  grasp_approach_offset_z_ = node_->declare_parameter<double>(
    "grasp_approach_offset_z", grasp_approach_offset_z_);
  post_grasp_lift_z_ = node_->declare_parameter<double>(
    "post_grasp_lift_z", post_grasp_lift_z_);
  gripper_grasp_width_ = node_->declare_parameter<double>(
    "gripper_grasp_width", gripper_grasp_width_);
  joint_state_wait_timeout_sec_ = node_->declare_parameter<double>(
    "joint_state_wait_timeout_sec", joint_state_wait_timeout_sec_);

  // Initialise TF2 listener before MoveIt so transforms are available early.
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // Create MoveIt interfaces once and reuse them in callbacks.
  // Group names come from the Panda SRDF: "panda_arm" for 7-DOF arm, "hand" for gripper.
  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");

  // Conservative defaults are easier to debug in coursework simulation.
  arm_group_->setPlanningTime(5.0);
  arm_group_->setNumPlanningAttempts(5);
  hand_group_->setPlanningTime(2.0);
  hand_group_->setNumPlanningAttempts(3);

  if (task2_capture_enabled_) {
    RCLCPP_INFO(
      node_->get_logger(),
      "Template capture mode enabled, output dir: %s",
      task2_capture_dir_.c_str());
  }

  RCLCPP_INFO(node_->get_logger(), "cw1 template class initialised with compatibility scaffold");
}

///////////////////////////////////////////////////////////////////////////////

geometry_msgs::msg::Quaternion
cw1::make_top_down_q() const
{
  // A simple top-down grasp attitude:
  tf2::Quaternion q;
  q.setRPY(3.14159265358979323846, 0.0, -0.78539816339744830962);
  q.normalize();
  return tf2::toMsg(q);
}

///////////////////////////////////////////////////////////////////////////////

std::string
cw1::identify_basket_colour(
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud,
  const geometry_msgs::msg::PointStamped &basket_loc)
{
  // Transform the basket world-frame position into the cloud's own frame so we
  // can search for nearby points directly in the cloud coordinate system.
  // Use stamp=0 (latest available) because basket_loc carries the wall-clock
  // time from the service request, which does not match Gazebo simulation time.
  geometry_msgs::msg::PointStamped basket_query = basket_loc;
  basket_query.header.stamp = rclcpp::Time(0);

  geometry_msgs::msg::PointStamped basket_in_cloud;
  try {
    basket_in_cloud = tf_buffer_->transform(
      basket_query,
      cloud->header.frame_id,
      tf2::durationFromSec(1.0));
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN(node_->get_logger(),
      "TF transform failed for basket at (%.3f, %.3f): %s",
      basket_loc.point.x, basket_loc.point.y, ex.what());
    return "none";
  }

  // Convert ROS PointCloud2 message to PCL point cloud for easier processing
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::fromROSMsg(*cloud, *pcl_cloud);

  // Search in the x-y plane of the cloud frame within a circle 
  // to find points belonging to the basket
  const float cx = static_cast<float>(basket_in_cloud.point.x);
  const float cy = static_cast<float>(basket_in_cloud.point.y);
  const float cz = static_cast<float>(basket_in_cloud.point.z);

  constexpr float kSearchRadius = 0.035f;
  constexpr float kSearchRadiusSq = kSearchRadius * kSearchRadius;
  constexpr float kZHalfWindow = 0.05f;

  size_t nearby_xyz_points = 0;
  size_t nonzero_rgb_points = 0;

  double sum_r = 0.0;
  double sum_g = 0.0;
  double sum_b = 0.0;

  // Iterate through all points in the point cloud to find colored points near the basket center
  for (const auto &pt : pcl_cloud->points) {
    // Skip invalid points (NaN or Inf)
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }

    const float dx = pt.x - cx;
    const float dy = pt.y - cy;
    const float dz = pt.z - cz;

    if (dx * dx + dy * dy > kSearchRadiusSq) {
      continue;
    }

    if (std::fabs(dz) > kZHalfWindow) {
      continue;
    }

    nearby_xyz_points++;

    if (pt.r == 0 && pt.g == 0 && pt.b == 0) {
      continue;
    }

    nonzero_rgb_points++;
    sum_r += static_cast<double>(pt.r);
    sum_g += static_cast<double>(pt.g);
    sum_b += static_cast<double>(pt.b);
  }

  if (nearby_xyz_points == 0 || nonzero_rgb_points == 0) {
    RCLCPP_WARN(node_->get_logger(),
      "Task 2: no usable coloured points found near basket at (%.3f, %.3f)",
      basket_loc.point.x, basket_loc.point.y);
    return "none";
  }

  // Calculate the average RGB values for the nearby valid points
  const double mean_r = sum_r / static_cast<double>(nonzero_rgb_points);
  const double mean_g = sum_g / static_cast<double>(nonzero_rgb_points);
  const double mean_b = sum_b / static_cast<double>(nonzero_rgb_points);

  std::string result = "none";

  // Classify the basket color based on the calculated mean RGB values using simple thresholds
  if (mean_r > 100.0 && mean_b > 100.0 && mean_g < 100.0) {
    result = "purple";
  } else if (mean_r > mean_b + 35.0 && mean_r > mean_g + 35.0 && mean_r > 100.0) {
    result = "red";
  } else if (mean_b > mean_r + 35.0 && mean_b > mean_g + 35.0 && mean_b > 100.0) {
    result = "blue";
  }

  RCLCPP_INFO(node_->get_logger(),
    "Task 2: basket at (%.3f, %.3f) mean RGB = (%.1f, %.1f, %.1f) -> %s",
    basket_loc.point.x, basket_loc.point.y, mean_r, mean_g, mean_b, result.c_str());

  return result;
}

///////////////////////////////////////////////////////////////////////////////

bool
cw1::move_arm_to_pose(const geometry_msgs::msg::Pose &target_pose)
{
  // Always start planning from current measured state to avoid unexpected trajectories.
  arm_group_->setStartStateToCurrentState();
  arm_group_->setPoseTarget(target_pose);
  arm_group_->setMaxVelocityScalingFactor(0.18);
  arm_group_->setMaxAccelerationScalingFactor(0.18);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const bool planning_ok =
    (arm_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!planning_ok) {
    arm_group_->clearPoseTargets();
    RCLCPP_WARN(node_->get_logger(), "Arm planning failed for requested pose");
    return false;
  }

  const bool execution_ok =
    (arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  arm_group_->clearPoseTargets();
  if (!execution_ok) {
    RCLCPP_WARN(node_->get_logger(), "Arm execution failed after successful planning");
    return false;
  }

  return true;
}

///////////////////////////////////////////////////////////////////////////////

bool
cw1::set_gripper_width(double width_m)
{
  // Panda hand limits are roughly [0.0, 0.08] total opening.
  // Clamp the requested width to avoid exceeding physical limits.
  const double clamped_width = std::clamp(width_m, 0.0, 0.08);
  const double finger_joint_target = clamped_width * 0.5;

  std::map<std::string, double> joint_targets;
  joint_targets["panda_finger_joint1"] = finger_joint_target;
  joint_targets["panda_finger_joint2"] = finger_joint_target;

  hand_group_->setStartStateToCurrentState();
  hand_group_->setJointValueTarget(joint_targets);

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const bool planning_ok =
    (hand_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!planning_ok) {
    RCLCPP_WARN(node_->get_logger(), "Gripper planning failed");
    return false;
  }

  const bool execution_ok =
    (hand_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!execution_ok) {
    RCLCPP_WARN(node_->get_logger(), "Gripper execution failed");
    return false;
  }

  return true;
}

bool
cw1::move_arm_linear_to(const geometry_msgs::msg::Pose &target_pose, double eef_step)
{
  // Plan and execute a linear Cartesian path to the target pose.
  // eef_step: distance between consecutive IK waypoints along the path.
  // Smaller = smoother path but more IK queries and higher failure rate.
  const double safe_eef_step = std::clamp(eef_step, 0.005, 0.02);
  // jump_threshold=0.0 disables joint-jump rejection. Safe here because
  // velocity/acceleration are already capped to 10%, preventing wild motion.
  const double jump_threshold = 0.0;

  // Cartesian path should start from the latest measured robot state.
  arm_group_->setStartStateToCurrentState();

  // Reduce execution aggressiveness; this helps avoid overshoot-like behavior in Gazebo.
  arm_group_->setMaxVelocityScalingFactor(0.18);
  arm_group_->setMaxAccelerationScalingFactor(0.18);

  // Build an explicit short segment from current pose to target pose.
  const auto current_pose_stamped = arm_group_->getCurrentPose();
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(current_pose_stamped.pose);
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory traj;
  const double fraction = arm_group_->computeCartesianPath(
    waypoints, safe_eef_step, jump_threshold, traj);

  if (fraction < 0.8) {
    RCLCPP_WARN(
      node_->get_logger(),
      "Cartesian path incomplete: fraction=%.3f eef_step=%.4f",
      fraction, safe_eef_step);
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  plan.trajectory_ = traj;
  const bool execution_ok =
    (arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!execution_ok) {
    RCLCPP_WARN(node_->get_logger(), "Cartesian execution failed");
    return false;
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////////
  /* Task 1*/
void
cw1::t1_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  // The service request gives:
  // - object_loc: centroid pose of the cube
  // - goal_loc: basket location point
  const auto object_pose = request->object_loc.pose;
  const auto goal_point = request->goal_loc.point;
  const auto top_down_q = make_top_down_q();

  // Build a deterministic sequence of waypoints:
  geometry_msgs::msg::Pose pre_grasp;
  pre_grasp.position = object_pose.position;
  pre_grasp.position.z += pick_offset_z_;
  pre_grasp.orientation = top_down_q;

  // grasp: descend vertically from pre_grasp.
  geometry_msgs::msg::Pose grasp = pre_grasp;
  grasp.position.z -= 0.24;

  // lift: return to pre_grasp height after grasping.
  geometry_msgs::msg::Pose lift = pre_grasp;

  // pre_place: safe high point directly above the basket.
  geometry_msgs::msg::Pose pre_place;
  pre_place.position.x = goal_point.x;
  pre_place.position.y = goal_point.y;
  pre_place.position.z = goal_point.z + place_offset_z_;
  pre_place.orientation = top_down_q;

  // Execute pick-and-place sequence.
  bool ok = true;
  ok = ok && set_gripper_width(0.07);                       // open gripper
  ok = ok && move_arm_linear_to(pre_grasp);                   // global plan: move above object
  ok = ok && move_arm_linear_to(grasp);                     // straight down to grasp height
  ok = ok && set_gripper_width(gripper_grasp_width_);       // close gripper
  ok = ok && move_arm_linear_to(lift);                      // straight up back to safe height
  ok = ok && move_arm_linear_to(pre_place);                   // global plan: transit to above basket
  ok = ok && set_gripper_width(0.07);                       // release cube
  // ok = ok && move_arm_linear_to(pre_grasp);                   // straight up out of basket

  if (ok) {
    RCLCPP_INFO(node_->get_logger(), "Task 1 execution finished successfully");
  } else {
    RCLCPP_WARN(node_->get_logger(), "Task 1 sequence finished with at least one failure");
  }
}

///////////////////////////////////////////////////////////////////////////////
  /* Task 2*/
void
cw1::t2_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task2Service::Response> response)
{
  RCLCPP_INFO(node_->get_logger(),
    "Task 2 started: %zu basket location(s) to check",
    request->basket_locs.size());

  // Observation height above the ground.  The wrist camera needs to be close
  // enough to resolve basket colours but far enough to keep the arm safe.
  constexpr double kObsZ = 0.50;

  for (const auto &basket_loc : request->basket_locs) {
    // 1. Move camera to a position directly above this basket 
    geometry_msgs::msg::Pose obs_pose;
    obs_pose.position.x = basket_loc.point.x;
    obs_pose.position.y = basket_loc.point.y;
    obs_pose.position.z = kObsZ;
    obs_pose.orientation = make_top_down_q();

    if (!move_arm_linear_to(obs_pose)) {
      RCLCPP_WARN(node_->get_logger(),
        "Could not reach observation pose for basket at (%.3f, %.3f) – marking none",
        basket_loc.point.x, basket_loc.point.y);
      response->basket_colours.push_back("none");
      continue;
    }

    // 2. Wait for the camera to settle and deliver a fresh frame
    // The sensor callback group runs on a separate thread (MultiThreadedExecutor),
    // so latest_cloud_ keeps updating while we sleep here.
    rclcpp::sleep_for(std::chrono::milliseconds(500));

    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
    {
      const auto deadline =
        node_->now() + rclcpp::Duration::from_seconds(3.0);
      while (rclcpp::ok()) {
        {
          std::lock_guard<std::mutex> lock(cloud_mutex_);
          if (latest_cloud_) {
            cloud = latest_cloud_;
          }
        }
        if (cloud) { break; }
        if (node_->now() > deadline) { break; }
        rclcpp::sleep_for(std::chrono::milliseconds(100));
      }
    }

    if (!cloud) {
      RCLCPP_ERROR(node_->get_logger(), "No point cloud received – marking none");
      response->basket_colours.push_back("none");
      continue;
    }

    // 3. Identify colour from the point cloud 
    const std::string colour = identify_basket_colour(cloud, basket_loc);
    response->basket_colours.push_back(colour);
  }

  for (size_t i = 0; i < response->basket_colours.size(); ++i) {
    RCLCPP_INFO(node_->get_logger(),
      "  basket_colours[%zu] = \"%s\"  (loc: %.3f, %.3f)",
      i,
      response->basket_colours[i].c_str(),
      request->basket_locs[i].point.x,
      request->basket_locs[i].point.y);
  }

  RCLCPP_INFO(node_->get_logger(), "Task 2 complete");
}

///////////////////////////////////////////////////////////////////////////////
  /* Task 3*/
void
cw1::t3_callback(
  const std::shared_ptr<cw1_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw1_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  (void)response;

  // Constants for cube dimensions
  constexpr double kCubeHeight = 0.04;
  constexpr double kCubeHalfHeight = kCubeHeight * 0.5;

  // Data structure to hold a single point with its classified seed color
  struct ClusterPoint
  {
    double x;
    double y;
    double z;
    uint8_t r;
    uint8_t g;
    uint8_t b;
    std::string seed_colour;
  };

  // Data structure to represent an object detected in a single point cloud frame
  struct RawDetection
  {
    std::string colour;
    geometry_msgs::msg::Point position;

    double size_x;
    double size_y;
    double size_z;

    double top_z;
    double centre_top_z;
    double rim_top_z;
    double rim_gap;
    double centre_fill_ratio;

    double mean_r;
    double mean_g;
    double mean_b;

    size_t point_count;
    size_t red_votes;
    size_t blue_votes;
    size_t purple_votes;
  };

  // Data structure to represent an object merged from multiple viewpoint detections
  struct MergedObject
  {
    std::string colour;
    geometry_msgs::msg::Point position;

    double size_x;
    double size_y;
    double size_z;

    double top_z;

    double max_rim_gap;
    double min_centre_fill_ratio;

    double mean_r;
    double mean_g;
    double mean_b;

    size_t point_count;
    size_t red_votes;
    size_t blue_votes;
    size_t purple_votes;

    size_t observations;
  };

  // Helper lambda to generate a top-down scanning pose for the camera
  auto make_scan_pose =
    [this](double x, double y, double z) -> geometry_msgs::msg::Pose
    {
      geometry_msgs::msg::Pose pose;
      pose.position.x = x;
      pose.position.y = y;
      pose.position.z = z;
      pose.orientation = make_top_down_q();
      return pose;
    };

  // Helper lambda to wait for and retrieve the latest point cloud message within a timeout
  auto get_latest_cloud_blocking =
    [this](double timeout_sec) -> sensor_msgs::msg::PointCloud2::ConstSharedPtr
    {
      sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
      const auto deadline =
        node_->now() + rclcpp::Duration::from_seconds(timeout_sec);

      while (rclcpp::ok()) {
        {
          std::lock_guard<std::mutex> lock(cloud_mutex_);
          if (latest_cloud_) {
            cloud = latest_cloud_;
          }
        }

        if (cloud) {
          return cloud;
        }

        if (node_->now() > deadline) {
          break;
        }

        rclcpp::sleep_for(std::chrono::milliseconds(100));
      }

      return nullptr;
    };

  // Helper lambda to convert RGB to HSV and classify color based on Hue
  auto classify_color_hsv = [](uint8_t r, uint8_t g, uint8_t b) -> std::string {
    // Convert RGB [0, 255] to [0, 1]
    double r_norm = r / 255.0;
    double g_norm = g / 255.0;
    double b_norm = b / 255.0;

    double cmax = std::max({r_norm, g_norm, b_norm});
    double cmin = std::min({r_norm, g_norm, b_norm});
    double delta = cmax - cmin;

    double h = 0.0; // Hue [0, 360)
    double s = 0.0; // Saturation [0, 1]
    double v = cmax; // Value [0, 1]

    if (delta > 0.0) {
      if (cmax == r_norm) {
        h = 60.0 * std::fmod(((g_norm - b_norm) / delta), 6.0);
      } else if (cmax == g_norm) {
        h = 60.0 * (((b_norm - r_norm) / delta) + 2.0);
      } else if (cmax == b_norm) {
        h = 60.0 * (((r_norm - g_norm) / delta) + 4.0);
      }
      if (h < 0.0) {
        h += 360.0;
      }
      s = delta / cmax;
    }

    // Filter out very dark or very colorless (gray/white) points
    if (v < 0.15 || s < 0.20) {
      return "none";
    }

    // Classify based on Hue angle
    // Red: ~0-20 or ~340-360
    // Blue: ~200-260
    // Purple: ~260-320
    if (h < 20.0 || h > 330.0) {
      return "red";
    } else if (h > 200.0 && h < 260.0) {
      return "blue";
    } else if (h >= 260.0 && h <= 330.0) {
      return "purple";
    }

    return "none";
  };

  // Helper lambda to classify a single point's color into predefined categories
  auto classify_seed_colour = [&classify_color_hsv](uint8_t r, uint8_t g, uint8_t b) -> std::string {
    return classify_color_hsv(r, g, b);
  };

  // Helper lambda to determine the dominant color of an object by looking only at its highest points
  // This helps avoid misclassification due to shadows or reflections on the lower parts
  auto dominant_colour_from_top_points =
    [&classify_color_hsv](const std::vector<ClusterPoint> &cluster, double top_z,
       size_t &red_votes, size_t &blue_votes, size_t &purple_votes,
       double &mean_r, double &mean_g, double &mean_b) -> std::string
    {
      red_votes = 0;
      blue_votes = 0;
      purple_votes = 0;
      mean_r = 0.0;
      mean_g = 0.0;
      mean_b = 0.0;

      size_t used = 0;
      // Define a vertical band (1.5 cm) from the highest point to sample colors from
      const double top_band = 0.015;

      for (const auto &p : cluster) {
        // Skip points that are below the top band
        if (p.z < top_z - top_band) {
          continue;
        }

        // Accumulate RGB values for averaging
        mean_r += static_cast<double>(p.r);
        mean_g += static_cast<double>(p.g);
        mean_b += static_cast<double>(p.b);
        used++;

        // Vote for the color based on HSV classification
        std::string color = classify_color_hsv(p.r, p.g, p.b);
        if (color == "purple") {
          purple_votes++;
        } else if (color == "red") {
          red_votes++;
        } else if (color == "blue") {
          blue_votes++;
        }
      }

      // Calculate mean RGB values for the top points
      if (used > 0) {
        mean_r /= static_cast<double>(used);
        mean_g /= static_cast<double>(used);
        mean_b /= static_cast<double>(used);
      }

      // Determine the final color based on the maximum number of votes
      if (red_votes == 0 && blue_votes == 0 && purple_votes == 0) {
        return "none";
      }
      if (purple_votes >= red_votes && purple_votes >= blue_votes) {
        return "purple";
      }
      if (red_votes >= blue_votes && red_votes >= purple_votes) {
        return "red";
      }
      return "blue";
    };

  // Core perception pipeline: extracts raw object detections from a single point cloud
  auto detect_raw_objects_from_cloud =
    [this, &classify_seed_colour, &dominant_colour_from_top_points](
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud)
      -> std::vector<RawDetection>
    {
      std::vector<RawDetection> objects;

      // Convert ROS PointCloud2 to PCL format
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::fromROSMsg(*cloud, *pcl_cloud);

      // Get the transform from the camera frame to the world frame
      geometry_msgs::msg::TransformStamped tf_cloud_to_world;
      try {
        tf_cloud_to_world = tf_buffer_->lookupTransform(
          "world",
          cloud->header.frame_id,
          rclcpp::Time(0),
          tf2::durationFromSec(1.0));
      } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN(
          node_->get_logger(),
          "Task 3: could not transform cloud to world: %s",
          ex.what());
        return objects;
      }

      // Convert the ROS transform to a TF2 transform for efficient point-by-point application
      tf2::Quaternion q(
        tf_cloud_to_world.transform.rotation.x,
        tf_cloud_to_world.transform.rotation.y,
        tf_cloud_to_world.transform.rotation.z,
        tf_cloud_to_world.transform.rotation.w);

      tf2::Vector3 t(
        tf_cloud_to_world.transform.translation.x,
        tf_cloud_to_world.transform.translation.y,
        tf_cloud_to_world.transform.translation.z);

      tf2::Transform tf_c2w(q, t);

      std::vector<ClusterPoint> pts;
      pts.reserve(pcl_cloud->points.size());

      // Filter and transform points
      for (const auto &pt : pcl_cloud->points) {
        // Skip invalid points
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
          continue;
        }

        // Only keep points that match one of our target colors
        const std::string seed_colour = classify_seed_colour(pt.r, pt.g, pt.b);
        if (seed_colour == "none") {
          continue;
        }

        // Transform point to world coordinates
        tf2::Vector3 p_cloud(pt.x, pt.y, pt.z);
        tf2::Vector3 p_world = tf_c2w * p_cloud;

        // Apply a workspace bounding box filter to ignore the robot and background
        if (p_world.x() < 0.20 || p_world.x() > 0.75) {
          continue;
        }
        if (p_world.y() < -0.45 || p_world.y() > 0.45) {
          continue;
        }
        if (p_world.z() < -0.02 || p_world.z() > 0.20) {
          continue;
        }

        // Store the valid, transformed point
        pts.push_back({
          p_world.x(), p_world.y(), p_world.z(),
          pt.r, pt.g, pt.b, seed_colour
        });
      }

      if (pts.empty()) {
        return objects;
      }

      // Perform Euclidean clustering using Breadth-First Search (BFS)
      std::vector<bool> visited(pts.size(), false);

      // Clustering parameters
      constexpr double kClusterRadius = 0.055; // Max xy distance between points in a cluster
      constexpr double kClusterRadiusSq = kClusterRadius * kClusterRadius;
      constexpr double kMaxDz = 0.06; // Max z distance between points in a cluster
      constexpr size_t kMinClusterSize = 18; // Minimum points to form a valid object

      for (size_t i = 0; i < pts.size(); ++i) {
        if (visited[i]) {
          continue;
        }

        visited[i] = true;
        std::queue<size_t> qidx;
        qidx.push(i);

        std::vector<size_t> cluster_indices;
        cluster_indices.push_back(i);

        // BFS to find all connected points of the same color
        while (!qidx.empty()) {
          const size_t cur = qidx.front();
          qidx.pop();

          for (size_t j = 0; j < pts.size(); ++j) {
            if (visited[j]) {
              continue;
            }

            // Only cluster points of the same color
            if (pts[j].seed_colour != pts[cur].seed_colour) {
              continue;
            }

            const double dx = pts[j].x - pts[cur].x;
            const double dy = pts[j].y - pts[cur].y;
            const double dz = std::fabs(pts[j].z - pts[cur].z);

            // Check if the point is within the clustering distance thresholds
            if ((dx * dx + dy * dy) < kClusterRadiusSq && dz < kMaxDz) {
              visited[j] = true;
              qidx.push(j);
              cluster_indices.push_back(j);
            }
          }
        }

        // Reject clusters that are too small (likely noise)
        if (cluster_indices.size() < kMinClusterSize) {
          continue;
        }

        // Process the valid cluster to extract object features
        std::vector<ClusterPoint> cluster;
        cluster.reserve(cluster_indices.size());

        // Bounding box variables
        double min_x = std::numeric_limits<double>::max();
        double min_y = std::numeric_limits<double>::max();
        double min_z = std::numeric_limits<double>::max();
        double max_x = -std::numeric_limits<double>::max();
        double max_y = -std::numeric_limits<double>::max();
        double max_z = -std::numeric_limits<double>::max();

        // Centroid accumulators
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;

        for (const auto idx : cluster_indices) {
          const auto &p = pts[idx];
          cluster.push_back(p);

          min_x = std::min(min_x, p.x);
          min_y = std::min(min_y, p.y);
          min_z = std::min(min_z, p.z);
          max_x = std::max(max_x, p.x);
          max_y = std::max(max_y, p.y);
          max_z = std::max(max_z, p.z);

          sum_x += p.x;
          sum_y += p.y;
          sum_z += p.z;
        }

        // Calculate cluster centroid
        const double cx = sum_x / static_cast<double>(cluster.size());
        const double cy = sum_y / static_cast<double>(cluster.size());
        const double cz = sum_z / static_cast<double>(cluster.size());

        // Calculate cluster dimensions
        const double size_x = max_x - min_x;
        const double size_y = max_y - min_y;
        const double size_z = max_z - min_z;
        const double max_xy = std::max(size_x, size_y);

        // Analyze the shape of the cluster to distinguish between cubes and baskets
        // Baskets have a hollow center and a higher rim, while cubes are solid
        const double half_extent = 0.5 * max_xy;
        const double inner_radius = std::max(0.012, 0.30 * half_extent);
        const double outer_r_min = std::max(0.015, 0.45 * half_extent);
        const double outer_r_max = std::max(0.025, 0.85 * half_extent);

        double centre_top_z = -std::numeric_limits<double>::max();
        double rim_top_z = -std::numeric_limits<double>::max();
        size_t centre_count = 0;
        size_t rim_count = 0;

        // Iterate through cluster points to find the highest points in the center and rim areas
        for (const auto &p : cluster) {
          const double dx = p.x - cx;
          const double dy = p.y - cy;
          const double rr = std::sqrt(dx * dx + dy * dy);

          if (rr <= inner_radius) {
            centre_count++;
            centre_top_z = std::max(centre_top_z, p.z);
          }

          if (rr >= outer_r_min && rr <= outer_r_max) {
            rim_count++;
            rim_top_z = std::max(rim_top_z, p.z);
          }
        }

        // If the center or rim heights were not updated, fallback to the maximum z of the cluster
        if (centre_top_z < -1e8) {
          centre_top_z = max_z;
        }
        if (rim_top_z < -1e8) {
          rim_top_z = max_z;
        }

        // Calculate shape features used for classification
        const double rim_gap = rim_top_z - centre_top_z;
        const double centre_fill_ratio =
          static_cast<double>(centre_count) / static_cast<double>(cluster.size());

        size_t red_votes = 0;
        size_t blue_votes = 0;
        size_t purple_votes = 0;
        double mean_r = 0.0;
        double mean_g = 0.0;
        double mean_b = 0.0;

        // Determine the final color of the object using only its top points
        const std::string final_colour =
          dominant_colour_from_top_points(
            cluster, max_z,
            red_votes, blue_votes, purple_votes,
            mean_r, mean_g, mean_b);

        if (final_colour == "none") {
          continue;
        }

        // Create a RawDetection object with all calculated features
        RawDetection obj;
        obj.colour = final_colour;
        obj.position.x = cx;
        obj.position.y = cy;
        obj.position.z = cz;

        obj.size_x = size_x;
        obj.size_y = size_y;
        obj.size_z = size_z;

        obj.top_z = max_z;
        obj.centre_top_z = centre_top_z;
        obj.rim_top_z = rim_top_z;
        obj.rim_gap = rim_gap;
        obj.centre_fill_ratio = centre_fill_ratio;

        obj.mean_r = mean_r;
        obj.mean_g = mean_g;
        obj.mean_b = mean_b;

        obj.point_count = cluster.size();
        obj.red_votes = red_votes;
        obj.blue_votes = blue_votes;
        obj.purple_votes = purple_votes;

        objects.push_back(obj);
      }

      return objects;
    };

  // Helper lambda to merge new raw detections into a persistent list of objects
  // This helps track objects across multiple camera frames and viewpoints
  auto merge_raw_detections =
    [](std::vector<MergedObject> &all_objects,
       const std::vector<RawDetection> &new_objects)
    {
      // Thresholds for considering two detections as the same object
      constexpr double kMergeXY = 0.09;
      constexpr double kMergeZ = 0.06;

      for (const auto &obj : new_objects) {
        bool merged = false;

        for (auto &existing : all_objects) {
          // Objects must have the same color to be merged
          if (existing.colour != obj.colour) {
            continue;
          }

          // Calculate spatial distance
          const double dx = existing.position.x - obj.position.x;
          const double dy = existing.position.y - obj.position.y;
          const double dz = std::fabs(existing.position.z - obj.position.z);
          const double dxy = std::sqrt(dx * dx + dy * dy);

          // If the new detection is close enough to an existing object, merge them
          if (dxy < kMergeXY && dz < kMergeZ) {
            const double n = static_cast<double>(existing.observations);

            // Update position using a running average
            existing.position.x =
              (existing.position.x * n + obj.position.x) / (n + 1.0);
            existing.position.y =
              (existing.position.y * n + obj.position.y) / (n + 1.0);
            existing.position.z =
              (existing.position.z * n + obj.position.z) / (n + 1.0);

            // Update bounding box sizes to the maximum observed
            existing.size_x = std::max(existing.size_x, obj.size_x);
            existing.size_y = std::max(existing.size_y, obj.size_y);
            existing.size_z = std::max(existing.size_z, obj.size_z);

            // Update the highest observed point
            existing.top_z = std::max(existing.top_z, obj.top_z);

            // Keep the most extreme shape features to help distinguish baskets
            existing.max_rim_gap = std::max(existing.max_rim_gap, obj.rim_gap);
            existing.min_centre_fill_ratio =
              std::min(existing.min_centre_fill_ratio, obj.centre_fill_ratio);

            // Update mean RGB using a running average
            existing.mean_r = (existing.mean_r * n + obj.mean_r) / (n + 1.0);
            existing.mean_g = (existing.mean_g * n + obj.mean_g) / (n + 1.0);
            existing.mean_b = (existing.mean_b * n + obj.mean_b) / (n + 1.0);

            // Accumulate point counts and color votes for majority voting
            existing.point_count = std::max(existing.point_count, obj.point_count);
            existing.red_votes += obj.red_votes;
            existing.blue_votes += obj.blue_votes;
            existing.purple_votes += obj.purple_votes;
            existing.observations++;

            merged = true;
            break;
          }
        }

        // If no matching object was found, create a new one
        if (!merged) {
          MergedObject mo;
          mo.colour = obj.colour;
          mo.position = obj.position;

          mo.size_x = obj.size_x;
          mo.size_y = obj.size_y;
          mo.size_z = obj.size_z;

          mo.top_z = obj.top_z;

          mo.max_rim_gap = obj.rim_gap;
          mo.min_centre_fill_ratio = obj.centre_fill_ratio;

          mo.mean_r = obj.mean_r;
          mo.mean_g = obj.mean_g;
          mo.mean_b = obj.mean_b;

          mo.point_count = obj.point_count;
          mo.red_votes = obj.red_votes;
          mo.blue_votes = obj.blue_votes;
          mo.purple_votes = obj.purple_votes;
          mo.observations = 1;

          all_objects.push_back(mo);
        }
      }
    };

  auto classify_final_type =
    [](const MergedObject &obj) -> std::string
    {
      const double min_xy = std::min(obj.size_x, obj.size_y);

      // if (min_xy < 0.070) {
      //   return "cube";
      // }
      if (min_xy > 0.08 && obj.size_z > 0.08) {
        return "basket";
      }
      if (obj.max_rim_gap > 0.05 && obj.min_centre_fill_ratio < 0.15) {
        return "basket";
      }
      return "cube";
    };

  auto execute_pick_and_place =
    [this](const geometry_msgs::msg::Point &cube_pt,
          const geometry_msgs::msg::Point &basket_pt) -> bool
    {
      const auto top_down_q = make_top_down_q();

      // Reuse the Task 1 motion pattern:
      // move above the cube, descend by a fixed calibrated amount,
      // lift back to the approach height, then move above the basket.
      geometry_msgs::msg::Pose pre_grasp;
      pre_grasp.position.x = cube_pt.x;
      pre_grasp.position.y = cube_pt.y;
      pre_grasp.position.z = cube_pt.z + pick_offset_z_;
      pre_grasp.orientation = top_down_q;

      geometry_msgs::msg::Pose grasp = pre_grasp;
      grasp.position.z = cube_pt.z + task3_pick_offset_z_;

      geometry_msgs::msg::Pose pre_place;
      pre_place.position.x = basket_pt.x;
      pre_place.position.y = basket_pt.y;
      // Lower the placement height slightly (by 6cm) to reduce the drop distance and prevent bouncing
      pre_place.position.z = basket_pt.z + place_offset_z_ - 0.06;
      pre_place.orientation = top_down_q;

      // geometry_msgs::msg::Pose retreat = pre_place;
      // retreat.position.z += 0.05;

      bool ok = true;
      ok = ok && set_gripper_width(0.07);                 // Open gripper.
      ok = ok && move_arm_linear_to(pre_grasp);             // Move above cube.
      ok = ok && move_arm_linear_to(grasp);               // Descend vertically.
      ok = ok && set_gripper_width(gripper_grasp_width_); // Close gripper.
      ok = ok && move_arm_linear_to(pre_grasp);                // Lift vertically.
      ok = ok && move_arm_linear_to(pre_place);             // Move above basket.
      
      // Wait for 0.5s to let the arm settle before releasing the cube
      if (ok) {
        rclcpp::sleep_for(std::chrono::milliseconds(500));
      }
      
      ok = ok && set_gripper_width(0.07);                 // Release cube.
      return ok;
    };

  RCLCPP_INFO(node_->get_logger(), "Task 3: scan stage started");

  // Define a set of predefined poses to scan the entire workspace.
  // These poses provide overlapping views to ensure all objects are detected.
  std::vector<geometry_msgs::msg::Pose> scan_poses;
  scan_poses.push_back(make_scan_pose(0.30, -0.30, 0.65));
  scan_poses.push_back(make_scan_pose(0.50, -0.30, 0.65));
  scan_poses.push_back(make_scan_pose(0.50,  0.00, 0.65));
  scan_poses.push_back(make_scan_pose(0.30,  0.00, 0.65));
  scan_poses.push_back(make_scan_pose(0.30,  0.30, 0.65));
  scan_poses.push_back(make_scan_pose(0.50,  0.30, 0.65));

  // Container to hold all unique objects found across all scan poses
  std::vector<MergedObject> all_objects;

  // Execute the scanning routine
  for (size_t i = 0; i < scan_poses.size(); ++i) {
    RCLCPP_INFO(node_->get_logger(), "Task 3: moving to scan pose %zu", i);

    // Move the arm to the next scanning viewpoint
    if (!move_arm_linear_to(scan_poses[i])) {
      RCLCPP_WARN(node_->get_logger(), "Task 3: failed to reach scan pose %zu", i);
      continue;
    }

    // Wait for the arm to settle and the camera image to stabilize
    rclcpp::sleep_for(std::chrono::milliseconds(700));

    // Capture multiple frames at each pose to improve detection robustness
    for (int frame_idx = 0; frame_idx < 4; ++frame_idx) {
      // Get the latest point cloud, blocking until one is available or timeout
      auto cloud = get_latest_cloud_blocking(1.5);
      if (!cloud) {
        continue;
      }

      // Process the point cloud to find objects
      const auto detections = detect_raw_objects_from_cloud(cloud);
      // Merge the new detections with the persistent list of objects
      merge_raw_detections(all_objects, detections);

      // Small delay between frames
      rclcpp::sleep_for(std::chrono::milliseconds(150));
    }
  }

  // Ensure we actually found something before proceeding
  if (all_objects.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Task 3: no objects detected");
    return;
  }

  // Data structure to hold the final, classified objects ready for manipulation
  struct FinalObject
  {
    std::string colour;
    std::string type; // "cube" or "basket"
    geometry_msgs::msg::Point position;
    double top_z;
  };

  std::vector<FinalObject> cubes;
  std::vector<FinalObject> baskets;

  RCLCPP_INFO(node_->get_logger(), "Task 3: final detections");
  // Classify each merged object as either a cube or a basket based on its shape features
  for (const auto &obj : all_objects) {
    const std::string type = classify_final_type(obj);

    // Log the properties of the detected object for debugging
    RCLCPP_INFO(
      node_->get_logger(),
      "  %s %s at (%.3f, %.3f, %.3f), top_z=%.3f, size=(%.3f, %.3f, %.3f), "
      "max_rim_gap=%.3f, min_centre_fill=%.3f, obs=%zu",
      obj.colour.c_str(),
      type.c_str(),
      obj.position.x,
      obj.position.y,
      obj.position.z,
      obj.top_z,
      obj.size_x,
      obj.size_y,
      obj.size_z,
      obj.max_rim_gap,
      obj.min_centre_fill_ratio,
      obj.observations);

    FinalObject fo;
    fo.colour = obj.colour;
    fo.type = type;
    fo.position = obj.position;
    fo.top_z = obj.top_z;

    // Separate cubes and baskets into different lists
    if (type == "cube") {
      cubes.push_back(fo);
    } else {
      baskets.push_back(fo);
    }
  }

  // Validate that we have both cubes to pick and baskets to place them in
  if (cubes.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Task 3: no cubes detected");
    return;
  }
  if (baskets.empty()) {
    RCLCPP_WARN(node_->get_logger(), "Task 3: no baskets detected");
    return;
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "Task 3: starting pick-and-place for %zu cube(s) and %zu basket(s)",
    cubes.size(), baskets.size());

  // Iterate through all detected cubes and attempt to place them in the matching basket
  for (const auto &cube : cubes) {
    int matched_idx = -1;

    // Find the basket that matches the color of the current cube
    for (size_t i = 0; i < baskets.size(); ++i) {
      if (baskets[i].colour == cube.colour) {
        matched_idx = static_cast<int>(i);
        break;
      }
    }

    // If no matching basket is found, skip this cube
    if (matched_idx < 0) {
      RCLCPP_WARN(
        node_->get_logger(),
        "Task 3: no basket found for %s cube at (%.3f, %.3f)",
        cube.colour.c_str(),
        cube.position.x,
        cube.position.y);
      continue;
    }

    // Calculate the precise pick point. 
    // The centroid is usually at the center of the visible points, 
    // but we want to grasp the cube slightly below its top surface.
    geometry_msgs::msg::Point cube_pick_point = cube.position;
    // Lower the grasp point slightly more to ensure a firmer grip on the cube
    cube_pick_point.z = cube.top_z - kCubeHalfHeight - 0.01;

      // The place point is the centroid of the basket
      geometry_msgs::msg::Point basket_place_point = baskets[matched_idx].position;

      // Execute the pick and place sequence
      const bool ok = execute_pick_and_place(
        cube_pick_point,
        basket_place_point);

      if (ok) {
      RCLCPP_INFO(
        node_->get_logger(),
        "Task 3: placed %s cube into %s basket",
        cube.colour.c_str(),
        baskets[matched_idx].colour.c_str());
    } else {
      RCLCPP_WARN(
        node_->get_logger(),
        "Task 3: pick-and-place failed for %s cube at (%.3f, %.3f)",
        cube.colour.c_str(),
        cube.position.x,
        cube.position.y);
    }
  }

  RCLCPP_INFO(node_->get_logger(), "Task 3 complete");
}
