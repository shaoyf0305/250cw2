/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

cw2::cw2(const rclcpp::Node::SharedPtr & node)
: node_(node),
  tf_buffer_(node->get_clock()),
  tf_listener_(tf_buffer_),
  g_cloud_ptr(new PointC)
{
  t1_service_ = node_->create_service<cw2_world_spawner::srv::Task1Service>(
    "/task1_start",
    std::bind(&cw2::t1_callback, this, std::placeholders::_1, std::placeholders::_2));
  t2_service_ = node_->create_service<cw2_world_spawner::srv::Task2Service>(
    "/task2_start",
    std::bind(&cw2::t2_callback, this, std::placeholders::_1, std::placeholders::_2));
  t3_service_ = node_->create_service<cw2_world_spawner::srv::Task3Service>(
    "/task3_start",
    std::bind(&cw2::t3_callback, this, std::placeholders::_1, std::placeholders::_2));

  pointcloud_topic_ = node_->declare_parameter<std::string>(
    "pointcloud_topic", "/r200/camera/depth_registered/points");
  pointcloud_qos_reliable_ =
    node_->declare_parameter<bool>("pointcloud_qos_reliable", true);

  pick_offset_z_ = node_->declare_parameter<double>("pick_offset_z", pick_offset_z_);
  grasp_descent_z_ = node_->declare_parameter<double>("grasp_descent_z", grasp_descent_z_);
  place_offset_z_ = node_->declare_parameter<double>("place_offset_z", place_offset_z_);
  gripper_open_width_ = node_->declare_parameter<double>("gripper_open_width", gripper_open_width_);
  gripper_grasp_width_ = node_->declare_parameter<double>("gripper_grasp_width", gripper_grasp_width_);
  cartesian_eef_step_ = node_->declare_parameter<double>("cartesian_eef_step", cartesian_eef_step_);
  cartesian_min_fraction_ =
    node_->declare_parameter<double>("cartesian_min_fraction", cartesian_min_fraction_);

  task1_apply_shape_xy_offset_ =
    node_->declare_parameter<bool>("task1_apply_shape_xy_offset", task1_apply_shape_xy_offset_);
  nought_grasp_offset_world_x_ = node_->declare_parameter<double>(
    "nought_grasp_offset_world_x", nought_grasp_offset_world_x_);
  nought_grasp_offset_world_y_ =
    node_->declare_parameter<double>("nought_grasp_offset_world_y", nought_grasp_offset_world_y_);
  cross_grasp_offset_world_x_ =
    node_->declare_parameter<double>("cross_grasp_offset_world_x", cross_grasp_offset_world_x_);
  cross_grasp_offset_world_y_ =
    node_->declare_parameter<double>("cross_grasp_offset_world_y", cross_grasp_offset_world_y_);
  nought_grasp_extra_yaw_rad_ =
    node_->declare_parameter<double>("nought_grasp_extra_yaw_rad", nought_grasp_extra_yaw_rad_);

  task2_obs_height_above_centroid_m_ = node_->declare_parameter<double>(
    "task2_obs_height_above_centroid_m", task2_obs_height_above_centroid_m_);
  task2_settle_ms_ = node_->declare_parameter<int>("task2_settle_ms", task2_settle_ms_);
  task2_inner_radius_m_ = node_->declare_parameter<double>("task2_inner_radius_m", task2_inner_radius_m_);
  task2_ring_r_min_m_ = node_->declare_parameter<double>("task2_ring_r_min_m", task2_ring_r_min_m_);
  task2_ring_r_max_m_ = node_->declare_parameter<double>("task2_ring_r_max_m", task2_ring_r_max_m_);
  task2_ground_sample_radius_m_ =
    node_->declare_parameter<double>("task2_ground_sample_radius_m", task2_ground_sample_radius_m_);
  task2_surface_above_ground_m_ =
    node_->declare_parameter<double>("task2_surface_above_ground_m", task2_surface_above_ground_m_);
  task2_inner_min_points_cross_ =
    node_->declare_parameter<int>("task2_inner_min_points_cross", task2_inner_min_points_cross_);
  task2_ring_inner_ratio_nought_min_ = node_->declare_parameter<double>(
    "task2_ring_inner_ratio_nought_min", task2_ring_inner_ratio_nought_min_);

  pointcloud_callback_group_ =
    node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  rclcpp::SubscriptionOptions pointcloud_sub_options;
  pointcloud_sub_options.callback_group = pointcloud_callback_group_;

  rclcpp::QoS pointcloud_qos = rclcpp::SensorDataQoS();
  if (pointcloud_qos_reliable_) {
    pointcloud_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
  }

  color_cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
    pointcloud_topic_,
    pointcloud_qos,
    std::bind(&cw2::cloud_callback, this, std::placeholders::_1),
    pointcloud_sub_options);

  arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "panda_arm");
  hand_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(node_, "hand");

  arm_group_->setPlanningTime(5.0);
  arm_group_->setNumPlanningAttempts(5);
  hand_group_->setPlanningTime(2.0);
  hand_group_->setNumPlanningAttempts(3);

  RCLCPP_INFO(
    node_->get_logger(),
    "cw2_team_36 template initialised with pointcloud topic '%s' (%s QoS)",
    pointcloud_topic_.c_str(),
    pointcloud_qos_reliable_ ? "reliable" : "sensor-data");
}

void cw2::cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
{
  pcl::PCLPointCloud2 pcl_cloud;
  pcl_conversions::toPCL(*msg, pcl_cloud);

  PointCPtr latest_cloud(new PointC);
  pcl::fromPCLPointCloud2(pcl_cloud, *latest_cloud);

  std::lock_guard<std::mutex> lock(cloud_mutex_);
  g_input_pc_frame_id_ = msg->header.frame_id;
  g_cloud_ptr = std::move(latest_cloud);
  ++g_cloud_sequence_;
}

geometry_msgs::msg::Quaternion
cw2::make_top_down_q() const
{
  tf2::Quaternion q;
  q.setRPY(3.14159265358979323846, 0.0, -0.78539816339744830962);
  q.normalize();
  geometry_msgs::msg::Quaternion msg;
  tf2::convert(q, msg);
  return msg;
}

geometry_msgs::msg::Quaternion
cw2::ee_orientation_for_shape(const std::string & shape) const
{
  tf2::Quaternion q;
  q.setRPY(3.14159265358979323846, 0.0, -0.78539816339744830962);
  q.normalize();
  if (shape == "nought") {
    const double yaw = nought_grasp_extra_yaw_rad_;
    if (std::abs(yaw) > 1e-9) {
      tf2::Quaternion qz;
      qz.setRPY(0.0, 0.0, yaw);
      q = q * qz;
      q.normalize();
    }
  }
  geometry_msgs::msg::Quaternion msg;
  tf2::convert(q, msg);
  return msg;
}

geometry_msgs::msg::Point
cw2::transform_point_to_planning_frame(
  const geometry_msgs::msg::Point & p, const std::string & source_frame)
{
  const std::string planning_frame = arm_group_->getPlanningFrame();
  // Gazebo poses are in `world`, but cw2_world_spawner labels PointStamped as `panda_link0`.
  // Try `world` first so we do not mis-treat world coordinates as base-frame coordinates.
  std::vector<std::string> frames;
  frames.push_back("world");
  if (!source_frame.empty() && source_frame != "world") {
    frames.push_back(source_frame);
  }

  for (const auto & frame : frames) {
    geometry_msgs::msg::PointStamped in;
    in.header.frame_id = frame;
    in.header.stamp = rclcpp::Time(0);
    in.point = p;
    try {
      geometry_msgs::msg::PointStamped out =
        tf_buffer_.transform(in, planning_frame, tf2::durationFromSec(2.0));
      return out.point;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_DEBUG(
        node_->get_logger(),
        "TF %s -> %s: %s", frame.c_str(), planning_frame.c_str(), ex.what());
    }
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "TF failed (world, %s); using untransformed point",
    source_frame.c_str());
  return p;
}

bool cw2::move_to_task2_observation_pose(const geometry_msgs::msg::PointStamped & object_loc)
{
  const std::string src =
    object_loc.header.frame_id.empty() ? "world" : object_loc.header.frame_id;
  const geometry_msgs::msg::Point p_plan =
    transform_point_to_planning_frame(object_loc.point, src);
  geometry_msgs::msg::Pose obs;
  obs.position.x = p_plan.x;
  obs.position.y = p_plan.y;
  obs.position.z = p_plan.z + task2_obs_height_above_centroid_m_;
  obs.orientation = make_top_down_q();
  if (move_arm_linear_to(obs, cartesian_eef_step_, "task2_obs")) {
    return true;
  }
  return move_arm_to_pose_joint(obs);
}

std::string cw2::classify_shape_from_cloud(
  const PointCPtr & cloud, const std::string & cloud_frame_id,
  const geometry_msgs::msg::PointStamped & object_loc)
{
  if (!cloud || cloud->empty() || cloud_frame_id.empty()) {
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task2] classify: FAIL (empty cloud or frame) -> unknown");
    return "unknown";
  }

  geometry_msgs::msg::TransformStamped tf_geom;
  try {
    tf_geom = tf_buffer_.lookupTransform(
      "world", cloud_frame_id, rclcpp::Time(0), tf2::durationFromSec(2.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(node_->get_logger(), "Task2 TF world<-cloud failed: %s", ex.what());
    return "unknown";
  }

  tf2::Transform T_w_c;
  tf2::fromMsg(tf_geom.transform, T_w_c);

  geometry_msgs::msg::PointStamped loc_world;
  loc_world.header.frame_id =
    object_loc.header.frame_id.empty() ? "world" : object_loc.header.frame_id;
  loc_world.header.stamp = rclcpp::Time(0);
  loc_world.point = object_loc.point;
  double ox = object_loc.point.x;
  double oy = object_loc.point.y;
  try {
    const auto lw = tf_buffer_.transform(loc_world, "world", tf2::durationFromSec(2.0));
    ox = lw.point.x;
    oy = lw.point.y;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_DEBUG(
      node_->get_logger(),
      "Task2 object point TF fallback (assume world coords): %s", ex.what());
  }

  const size_t n = cloud->size();
  const size_t stride = std::max<size_t>(1, n / 50000);

  std::vector<float> ground_z_samples;
  ground_z_samples.reserve(n / stride + 1);
  for (size_t i = 0; i < n; i += stride) {
    const auto & pt = cloud->points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    const tf2::Vector3 vc(pt.x, pt.y, pt.z);
    const tf2::Vector3 vw = T_w_c * vc;
    const double dh =
      std::hypot(vw.x() - ox, vw.y() - oy);
    if (dh < task2_ground_sample_radius_m_) {
      ground_z_samples.push_back(static_cast<float>(vw.z()));
    }
  }
  if (ground_z_samples.empty()) {
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task2] classify: FAIL (no ground samples near object) -> unknown");
    return "unknown";
  }
  std::sort(ground_z_samples.begin(), ground_z_samples.end());
  const size_t gi =
    std::min(static_cast<size_t>(ground_z_samples.size() * 0.06), ground_z_samples.size() - 1);
  const float ground_z = ground_z_samples[gi];
  const float z_obj_min = ground_z + static_cast<float>(task2_surface_above_ground_m_);

  std::int64_t inner = 0;
  std::int64_t ring = 0;
  for (size_t i = 0; i < n; i += stride) {
    const auto & pt = cloud->points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    const tf2::Vector3 vc(pt.x, pt.y, pt.z);
    const tf2::Vector3 vw = T_w_c * vc;
    if (static_cast<float>(vw.z()) < z_obj_min) {
      continue;
    }
    const double dh = std::hypot(vw.x() - ox, vw.y() - oy);
    if (dh < task2_inner_radius_m_) {
      ++inner;
    } else if (dh >= task2_ring_r_min_m_ && dh <= task2_ring_r_max_m_) {
      ++ring;
    }
  }

  const double ratio =
    static_cast<double>(ring) / static_cast<double>(inner + 8);

  std::string label;
  if (inner >= task2_inner_min_points_cross_) {
    label = "cross";
  } else if (ratio >= task2_ring_inner_ratio_nought_min_) {
    label = "nought";
  } else {
    label = "cross";
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "[Task2] >>> classified '%s' <<<  | inner=%lld  ring=%lld  ring/(inner+8)=%.2f  | "
    "ground_z≈%.3f  obj_z>=%.3f  @(%.3f,%.3f)",
    label.c_str(),
    static_cast<long long>(inner),
    static_cast<long long>(ring),
    ratio,
    static_cast<double>(ground_z),
    static_cast<double>(z_obj_min),
    ox, oy);
  return label;
}

bool cw2::move_arm_to_pose_joint(const geometry_msgs::msg::Pose & target_pose)
{
  arm_group_->setStartStateToCurrentState();
  arm_group_->setPoseTarget(target_pose);
  arm_group_->setMaxVelocityScalingFactor(0.22);
  arm_group_->setMaxAccelerationScalingFactor(0.22);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const bool planning_ok =
    (arm_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  if (!planning_ok) {
    arm_group_->clearPoseTargets();
    RCLCPP_WARN(node_->get_logger(), "Joint-space arm planning failed");
    return false;
  }

  const bool execution_ok =
    (arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
  arm_group_->clearPoseTargets();
  if (!execution_ok) {
    RCLCPP_WARN(node_->get_logger(), "Joint-space arm execution failed");
    return false;
  }

  return true;
}

bool cw2::set_gripper_width(double width_m)
{
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

bool cw2::move_arm_linear_to(
  const geometry_msgs::msg::Pose & target_pose, double eef_step, const char * step_name)
{
  const double safe_eef_step = std::clamp(eef_step, 0.005, 0.02);
  const double jump_threshold = 0.0;
  const char * label = (step_name && step_name[0]) ? step_name : "move";

  arm_group_->setStartStateToCurrentState();
  arm_group_->setMaxVelocityScalingFactor(0.22);
  arm_group_->setMaxAccelerationScalingFactor(0.22);

  const auto current_pose_stamped = arm_group_->getCurrentPose();
  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(current_pose_stamped.pose);
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory traj;
  const double fraction = arm_group_->computeCartesianPath(
    waypoints, safe_eef_step, jump_threshold, traj);

  if (fraction + 1e-6 >= cartesian_min_fraction_) {
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = traj;
    const bool execution_ok =
      (arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
    if (!execution_ok) {
      RCLCPP_WARN(
        node_->get_logger(),
        "[%s] Cartesian execute failed; trying joint motion", label);
      return move_arm_to_pose_joint(target_pose);
    }
    return true;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "[%s] Cartesian incomplete (fraction=%.3f, need %.2f); using joint motion",
    label, fraction, cartesian_min_fraction_);
  return move_arm_to_pose_joint(target_pose);
}

void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  const std::string & shape = request->shape_type;
  RCLCPP_INFO(
    node_->get_logger(),
    "Task 1: shape_type=%s, object=(%.3f,%.3f,%.3f) goal=(%.3f,%.3f,%.3f) [header=%s]",
    shape.c_str(),
    request->object_point.point.x,
    request->object_point.point.y,
    request->object_point.point.z,
    request->goal_point.point.x,
    request->goal_point.point.y,
    request->goal_point.point.z,
    request->object_point.header.frame_id.c_str());

  // Gazebo reports poses in `world`; the spawner labels PointStamped as `panda_link0`.
  // Try the declared frame first, then fall back inside transform_point_to_planning_frame().
  const std::string src_frame = request->object_point.header.frame_id.empty() ?
    "world" : request->object_point.header.frame_id;

  double off_x = 0.0;
  double off_y = 0.0;
  if (task1_apply_shape_xy_offset_) {
    if (shape == "nought") {
      off_x = nought_grasp_offset_world_x_;
      off_y = nought_grasp_offset_world_y_;
    } else if (shape == "cross") {
      off_x = cross_grasp_offset_world_x_;
      off_y = cross_grasp_offset_world_y_;
    }
  }

  geometry_msgs::msg::Point obj_world = request->object_point.point;
  obj_world.x += off_x;
  obj_world.y += off_y;

  geometry_msgs::msg::Point goal_world = request->goal_point.point;
  goal_world.x += off_x;
  goal_world.y += off_y;

  if (task1_apply_shape_xy_offset_ && (off_x != 0.0 || off_y != 0.0)) {
    RCLCPP_INFO(
      node_->get_logger(),
      "Task1: same world XY offset (%.3f, %.3f) m for pick and place (object vs basket centre)",
      off_x, off_y);
  }

  geometry_msgs::msg::Point obj_p = transform_point_to_planning_frame(obj_world, src_frame);
  const std::string goal_src = request->goal_point.header.frame_id.empty() ?
    "world" : request->goal_point.header.frame_id;
  geometry_msgs::msg::Point goal_p = transform_point_to_planning_frame(goal_world, goal_src);

  const auto top_down_q = ee_orientation_for_shape(shape);

  geometry_msgs::msg::Pose pre_grasp;
  pre_grasp.position = obj_p;
  pre_grasp.position.z += pick_offset_z_;
  pre_grasp.orientation = top_down_q;

  geometry_msgs::msg::Pose grasp = pre_grasp;
  grasp.position.z -= grasp_descent_z_;

  // After grasp, lift to the same absolute height used above the basket (not pick_offset_z_).
  geometry_msgs::msg::Pose lift;
  lift.position.x = pre_grasp.position.x;
  lift.position.y = pre_grasp.position.y;
  lift.position.z = goal_p.z + place_offset_z_;
  lift.orientation = top_down_q;

  geometry_msgs::msg::Pose pre_place;
  pre_place.position.x = goal_p.x;
  pre_place.position.y = goal_p.y;
  pre_place.position.z = goal_p.z + place_offset_z_;
  pre_place.orientation = top_down_q;

  const double grasp_w = gripper_grasp_width_;

  bool ok = true;
  ok = ok && set_gripper_width(gripper_open_width_);
  ok = ok && move_arm_linear_to(pre_grasp, cartesian_eef_step_, "pre_grasp");
  ok = ok && move_arm_linear_to(grasp, cartesian_eef_step_, "grasp");
  ok = ok && set_gripper_width(grasp_w);
  ok = ok && move_arm_linear_to(lift, cartesian_eef_step_, "lift");
  ok = ok && move_arm_linear_to(pre_place, cartesian_eef_step_, "pre_place");
  ok = ok && set_gripper_width(gripper_open_width_);

  if (ok) {
    RCLCPP_INFO(node_->get_logger(), "Task 1 finished successfully");
  } else {
    RCLCPP_WARN(node_->get_logger(), "Task 1 finished with at least one failure");
  }
}

void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  response->mystery_object_num = -1;

  RCLCPP_WARN(
    node_->get_logger(),
    "\n"
    "****************************************************************\n"
    "*  TASK 2 — shape matching (nought vs cross)                  *\n"
    "****************************************************************");

  if (request->ref_object_points.size() < 2) {
    RCLCPP_ERROR(node_->get_logger(), "Task2: need two reference points");
    return;
  }

  auto capture_cloud = [this]() -> PointCPtr {
    const auto deadline = node_->now() + rclcpp::Duration::from_seconds(3.0);
    while (rclcpp::ok()) {
      PointCPtr cloud;
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        cloud = g_cloud_ptr;
      }
      if (cloud && !cloud->empty()) {
        return cloud;
      }
      if (node_->now() > deadline) {
        break;
      }
      rclcpp::sleep_for(std::chrono::milliseconds(100));
    }
    return nullptr;
  };

  auto observe_and_classify =
    [this, &capture_cloud](const geometry_msgs::msg::PointStamped & loc) -> std::string {
      if (!move_to_task2_observation_pose(loc)) {
        RCLCPP_WARN(
          node_->get_logger(),
          "[Task2] !!! could NOT reach observation pose (%.3f, %.3f) !!!",
          loc.point.x, loc.point.y);
        return "unknown";
      }
      rclcpp::sleep_for(std::chrono::milliseconds(static_cast<int>(task2_settle_ms_)));

      PointCPtr cloud;
      std::string frame_id;
      {
        std::lock_guard<std::mutex> lock(cloud_mutex_);
        cloud = g_cloud_ptr;
        frame_id = g_input_pc_frame_id_;
      }
      if (!cloud || cloud->empty()) {
        cloud = capture_cloud();
      }
      if (!cloud || cloud->empty()) {
        RCLCPP_WARN(node_->get_logger(), "[Task2] !!! no point cloud !!!");
        return "unknown";
      }
      return classify_shape_from_cloud(cloud, frame_id, loc);
    };

  RCLCPP_WARN(node_->get_logger(), "[Task2] --- reference [0] ---");
  const std::string s0 = observe_and_classify(request->ref_object_points[0]);
  RCLCPP_WARN(node_->get_logger(), "[Task2] --- reference [1] ---");
  const std::string s1 = observe_and_classify(request->ref_object_points[1]);
  RCLCPP_WARN(node_->get_logger(), "[Task2] --- MYSTERY ---");
  const std::string sm = observe_and_classify(request->mystery_object_point);

  if (sm != "unknown") {
    if (sm == s0 && sm != s1) {
      response->mystery_object_num = 1;
    } else if (sm == s1 && sm != s0) {
      response->mystery_object_num = 2;
    } else if (sm == s0 && sm == s1) {
      response->mystery_object_num = 1;
      RCLCPP_WARN(
        node_->get_logger(),
        "[Task2] mystery matches BOTH references (ambiguous) -> default answer 1");
    }
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "\n"
    "================================================================\n"
    "  TASK 2  SUMMARY\n"
    "    reference 1:   %s\n"
    "    reference 2:   %s\n"
    "    mystery shape: %s\n"
    "    mystery_object_num = %lld  (%s)\n"
    "================================================================\n",
    s0.c_str(),
    s1.c_str(),
    sm.c_str(),
    static_cast<long long>(response->mystery_object_num),
    response->mystery_object_num == 1 ? "mystery matches reference 1" :
    response->mystery_object_num == 2 ? "mystery matches reference 2" :
    "FAILED or unknown (-1)");
}

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();

  std::string frame_id;
  std::size_t point_count = 0;
  std::uint64_t sequence = 0;
  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    frame_id = g_input_pc_frame_id_;
    point_count = g_cloud_ptr ? g_cloud_ptr->size() : 0;
    sequence = g_cloud_sequence_;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "Task 3 is not implemented in cw2_team_36. Latest cloud: seq=%llu frame='%s' points=%zu",
    static_cast<unsigned long long>(sequence),
    frame_id.c_str(),
    point_count);
}
