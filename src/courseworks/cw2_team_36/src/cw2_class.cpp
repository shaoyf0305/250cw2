/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#include <cw2_class.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
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


namespace
{

double wrap_half_pi(double yaw)
{
  while (yaw >= M_PI_2) {
    yaw -= M_PI;
  }
  while (yaw < -M_PI_2) {
    yaw += M_PI;
  }
  return yaw;
}

bool collect_task1_object_top_points(
  const tf2_ros::Buffer & tf_buffer,
  const rclcpp::Logger & logger,
  const PointCPtr & cloud,
  const std::string & cloud_frame_id,
  const geometry_msgs::msg::PointStamped & object_loc,
  std::vector<std::pair<double, double>> & xy_pts,
  double & ox,
  double & oy,
  double & ground_z)
{
  xy_pts.clear();
  ox = object_loc.point.x;
  oy = object_loc.point.y;
  ground_z = 0.0;

  if (!cloud || cloud->empty() || cloud_frame_id.empty()) {
    RCLCPP_WARN(logger, "[Task1] yaw estimate failed: empty cloud/frame");
    return false;
  }

  geometry_msgs::msg::TransformStamped tf_geom;
  try {
    tf_geom = tf_buffer.lookupTransform(
      "world", cloud_frame_id, rclcpp::Time(0), tf2::durationFromSec(2.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(logger, "[Task1] TF world<-cloud failed: %s", ex.what());
    return false;
  }

  tf2::Transform T_w_c;
  tf2::fromMsg(tf_geom.transform, T_w_c);

  geometry_msgs::msg::PointStamped loc_world;
  loc_world.header.frame_id =
    object_loc.header.frame_id.empty() ? "world" : object_loc.header.frame_id;
  loc_world.header.stamp = rclcpp::Time(0);
  loc_world.point = object_loc.point;

  try {
    const auto lw = tf_buffer.transform(loc_world, "world", tf2::durationFromSec(2.0));
    ox = lw.point.x;
    oy = lw.point.y;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_DEBUG(
      logger,
      "[Task1] object point TF fallback (assume world coords): %s", ex.what());
  }

  const size_t n = cloud->size();
  const size_t stride = std::max<size_t>(1, n / 50000);

  std::vector<float> ground_z_samples;
  ground_z_samples.reserve(n / stride + 1);

  constexpr double k_ground_sample_radius = 0.14;
  for (size_t i = 0; i < n; i += stride) {
    const auto & pt = cloud->points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }

    const tf2::Vector3 vc(pt.x, pt.y, pt.z);
    const tf2::Vector3 vw = T_w_c * vc;
    const double dh = std::hypot(vw.x() - ox, vw.y() - oy);
    if (dh < k_ground_sample_radius) {
      ground_z_samples.push_back(static_cast<float>(vw.z()));
    }
  }

  if (ground_z_samples.size() < 20) {
    RCLCPP_WARN(logger, "[Task1] yaw estimate failed: not enough ground samples");
    return false;
  }

  std::sort(ground_z_samples.begin(), ground_z_samples.end());
  const size_t gi =
    std::min(static_cast<size_t>(ground_z_samples.size() * 0.06), ground_z_samples.size() - 1);
  ground_z = static_cast<double>(ground_z_samples[gi]);
  const float z_min = static_cast<float>(ground_z) + 0.012f;

  constexpr double k_object_crop_radius = 0.11;
  xy_pts.reserve(n / stride + 1);
  for (size_t i = 0; i < n; i += stride) {
    const auto & pt = cloud->points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }

    const tf2::Vector3 vc(pt.x, pt.y, pt.z);
    const tf2::Vector3 vw = T_w_c * vc;
    if (static_cast<float>(vw.z()) < z_min) {
      continue;
    }

    const double dx = vw.x() - ox;
    const double dy = vw.y() - oy;
    const double dh = std::hypot(dx, dy);
    if (dh <= k_object_crop_radius) {
      xy_pts.emplace_back(dx, dy);
    }
  }

  if (xy_pts.size() < 40) {
    RCLCPP_WARN(logger, "[Task1] yaw estimate failed: not enough object points");
    return false;
  }

  return true;
}

bool estimate_cross_pca_yaw_impl(
  const rclcpp::Logger & logger,
  const std::vector<std::pair<double, double>> & xy_pts,
  double & estimated_yaw_rad)
{
  estimated_yaw_rad = 0.0;

  double mean_x = 0.0;
  double mean_y = 0.0;
  for (const auto & p : xy_pts) {
    mean_x += p.first;
    mean_y += p.second;
  }
  mean_x /= static_cast<double>(xy_pts.size());
  mean_y /= static_cast<double>(xy_pts.size());

  double sxx = 0.0;
  double syy = 0.0;
  double sxy = 0.0;
  for (const auto & p : xy_pts) {
    const double dx = p.first - mean_x;
    const double dy = p.second - mean_y;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }

  if (xy_pts.size() > 1) {
    const double denom = static_cast<double>(xy_pts.size() - 1);
    sxx /= denom;
    syy /= denom;
    sxy /= denom;
  }

  const double yaw = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
  if (!std::isfinite(yaw)) {
    RCLCPP_WARN(logger, "[Task1] yaw estimate failed: non-finite yaw");
    return false;
  }

  estimated_yaw_rad = wrap_half_pi(yaw);
  return true;
}

bool estimate_nought_inner_edge_yaw_impl(
  const rclcpp::Logger & logger,
  const std::vector<std::pair<double, double>> & xy_pts,
  double & estimated_yaw_rad)
{
  estimated_yaw_rad = 0.0;

  if (xy_pts.size() < 60) {
    RCLCPP_WARN(logger, "[Task1] nought yaw estimate failed: not enough object points");
    return false;
  }

  constexpr int k_angle_bins = 180;
  std::vector<double> min_radius(k_angle_bins, std::numeric_limits<double>::infinity());

  for (const auto & p : xy_pts) {
    const double r = std::hypot(p.first, p.second);
    if (!std::isfinite(r) || r < 1e-4) {
      continue;
    }

    double ang = std::atan2(p.second, p.first);
    if (ang < 0.0) {
      ang += 2.0 * M_PI;
    }
    const int idx = std::min(
      k_angle_bins - 1,
      static_cast<int>(std::floor((ang / (2.0 * M_PI)) * static_cast<double>(k_angle_bins))));
    min_radius[idx] = std::min(min_radius[idx], r);
  }

  bool have_any_bin = false;
  for (double v : min_radius) {
    if (std::isfinite(v)) {
      have_any_bin = true;
      break;
    }
  }
  if (!have_any_bin) {
    RCLCPP_WARN(logger, "[Task1] nought yaw estimate failed: no radial profile");
    return false;
  }

  for (int pass = 0; pass < 2; ++pass) {
    std::vector<double> filled = min_radius;
    for (int i = 0; i < k_angle_bins; ++i) {
      if (std::isfinite(filled[i])) {
        continue;
      }
      for (int step = 1; step < k_angle_bins; ++step) {
        const int left = (i - step + k_angle_bins) % k_angle_bins;
        const int right = (i + step) % k_angle_bins;
        if (std::isfinite(min_radius[left])) {
          filled[i] = min_radius[left];
          break;
        }
        if (std::isfinite(min_radius[right])) {
          filled[i] = min_radius[right];
          break;
        }
      }
    }
    min_radius.swap(filled);
  }

  double best_score = std::numeric_limits<double>::infinity();
  int best_idx = 0;
  const int quarter_turn = k_angle_bins / 4;
  for (int i = 0; i < quarter_turn; ++i) {
    const int i90 = (i + quarter_turn) % k_angle_bins;
    const int i180 = (i + 2 * quarter_turn) % k_angle_bins;
    const int i270 = (i + 3 * quarter_turn) % k_angle_bins;

    const double score =
      min_radius[i] + min_radius[i90] + min_radius[i180] + min_radius[i270];
    if (score < best_score) {
      best_score = score;
      best_idx = i;
    }
  }

  const double normal_yaw =
    (static_cast<double>(best_idx) + 0.5) * (2.0 * M_PI / static_cast<double>(k_angle_bins));
  const double edge_yaw = normal_yaw + M_PI_2;
  estimated_yaw_rad = wrap_half_pi(edge_yaw);

  return true;
}

}  // namespace

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
    "cw2_team_36 rotation-aware version initialised with pointcloud topic '%s' (%s QoS)",
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

geometry_msgs::msg::Quaternion
cw2::ee_orientation_for_shape_with_world_yaw(
  const std::string & shape,
  double world_yaw_rad) const
{
  const double base_yaw = -0.78539816339744830962;  // -pi/4
  double final_yaw = base_yaw;

  if (shape == "cross") {
    final_yaw += world_yaw_rad;
  } else if (shape == "nought") {
    // The nought also benefits from an observed principal yaw because the grasp
    // offset should be applied relative to the visible edge, not only in a
    // fixed world direction.
    final_yaw += world_yaw_rad + nought_grasp_extra_yaw_rad_;
  }

  tf2::Quaternion q;
  q.setRPY(3.14159265358979323846, 0.0, final_yaw);
  q.normalize();

  geometry_msgs::msg::Quaternion msg;
  tf2::convert(q, msg);
  return msg;
}

geometry_msgs::msg::Point
cw2::transform_point_to_planning_frame(
  const geometry_msgs::msg::Point & p, const std::string & source_frame)
{
  const std::string planning_frame = arm_group_->getPlanningFrame();
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

bool cw2::capture_latest_cloud(PointCPtr & cloud, std::string & frame_id, double timeout_sec)
{
  const auto deadline = node_->now() + rclcpp::Duration::from_seconds(timeout_sec);
  std::uint64_t start_seq = 0;

  {
    std::lock_guard<std::mutex> lock(cloud_mutex_);
    start_seq = g_cloud_sequence_;
    if (g_cloud_ptr && !g_cloud_ptr->empty() && !g_input_pc_frame_id_.empty()) {
      cloud = g_cloud_ptr;
      frame_id = g_input_pc_frame_id_;
      return true;
    }
  }

  while (rclcpp::ok() && node_->now() < deadline) {
    {
      std::lock_guard<std::mutex> lock(cloud_mutex_);
      if (g_cloud_ptr && !g_cloud_ptr->empty() &&
        !g_input_pc_frame_id_.empty() &&
        g_cloud_sequence_ >= start_seq)
      {
        cloud = g_cloud_ptr;
        frame_id = g_input_pc_frame_id_;
        return true;
      }
    }
    rclcpp::sleep_for(std::chrono::milliseconds(60));
  }

  return false;
}

bool cw2::estimate_object_yaw_from_cloud(
  const PointCPtr & cloud,
  const std::string & cloud_frame_id,
  const geometry_msgs::msg::PointStamped & object_loc,
  double & estimated_yaw_rad) const
{
  std::vector<std::pair<double, double>> xy_pts;
  double ox = 0.0;
  double oy = 0.0;
  double ground_z = 0.0;
  if (!collect_task1_object_top_points(
      tf_buffer_, node_->get_logger(), cloud, cloud_frame_id, object_loc, xy_pts, ox, oy, ground_z))
  {
    estimated_yaw_rad = 0.0;
    return false;
  }

  if (!estimate_cross_pca_yaw_impl(node_->get_logger(), xy_pts, estimated_yaw_rad)) {
    return false;
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "[Task1] estimated object yaw = %.3f rad (%.1f deg), samples=%zu, ground_z=%.3f",
    estimated_yaw_rad,
    estimated_yaw_rad * 180.0 / M_PI,
    xy_pts.size(),
    ground_z);

  return true;
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
  arm_group_->setMaxVelocityScalingFactor(0.16);
  arm_group_->setMaxAccelerationScalingFactor(0.16);
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
  arm_group_->setMaxVelocityScalingFactor(0.16);
  arm_group_->setMaxAccelerationScalingFactor(0.16);

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

  const std::string src_frame = request->object_point.header.frame_id.empty() ?
    "world" : request->object_point.header.frame_id;
  const std::string goal_src = request->goal_point.header.frame_id.empty() ?
    "world" : request->goal_point.header.frame_id;

  // Use the service-provided centroids without any grasp offset to build the
  // initial observation route. The offset is applied only after yaw has been
  // estimated from the observation pose.
  const geometry_msgs::msg::Point nominal_obj_p =
    transform_point_to_planning_frame(request->object_point.point, src_frame);

  const double k_task1_safe_travel_z = 0.6;
  const geometry_msgs::msg::Quaternion top_down_q = make_top_down_q();
  // const geometry_msgs::msg::Pose current_pose = arm_group_->getCurrentPose().pose;

  // Step 1: move straight upward from the current pose to the fixed travel
  // height. This keeps the first segment simple and vertical.
  // geometry_msgs::msg::Pose safe_raise_pose = current_pose;
  // safe_raise_pose.position.x = nominal_obj_p.x;
  // safe_raise_pose.position.y = nominal_obj_p.y;
  // safe_raise_pose.position.z = k_task1_safe_travel_z;

  // Step 2: from the raised transit pose, move to a fixed top-down observation
  // pose directly above the object at the same travel height.
  geometry_msgs::msg::Pose observation_pose;
  observation_pose.position.x = nominal_obj_p.x;
  observation_pose.position.y = nominal_obj_p.y;
  observation_pose.position.z = k_task1_safe_travel_z;
  observation_pose.orientation = top_down_q;

  auto execute_cartesian_to_target =
    [this](
      const geometry_msgs::msg::Pose & target_pose,
      double eef_step,
      const char * step_name) -> bool
    {
      const double safe_eef_step = std::clamp(eef_step, 0.005, 0.02);
      const double jump_threshold = 0.0;
      const char * label = (step_name && step_name[0]) ? step_name : "move";

      arm_group_->setStartStateToCurrentState();
      arm_group_->setMaxVelocityScalingFactor(0.16);
      arm_group_->setMaxAccelerationScalingFactor(0.16);

      std::vector<geometry_msgs::msg::Pose> waypoints;
      // Start from the measured current pose to avoid revisiting an outdated
      // nominal start pose after grasping or after a previous joint-space move.
      waypoints.push_back(target_pose);

      moveit_msgs::msg::RobotTrajectory traj;
      const double fraction = arm_group_->computeCartesianPath(
        waypoints, safe_eef_step, jump_threshold, traj);

      if (fraction + 1e-6 < cartesian_min_fraction_) {
        RCLCPP_WARN(
          node_->get_logger(),
          "[%s] Cartesian incomplete (fraction=%.3f, need %.2f)",
          label, fraction, cartesian_min_fraction_);
        return false;
      }

      moveit::planning_interface::MoveGroupInterface::Plan plan;
      plan.trajectory_ = traj;
      const bool execution_ok =
        (arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);
      if (!execution_ok) {
        RCLCPP_WARN(node_->get_logger(), "[%s] Cartesian execution failed", label);
        return false;
      }

      return true;
    };

  bool ok = true;
  ok = ok && set_gripper_width(gripper_open_width_);

  bool reached_observation = true;
  // if (safe_raise_pose.position.z > current_pose.position.z + 1e-4) {
    // reached_observation = reached_observation &&
    //   execute_cartesian_to_target(safe_raise_pose, cartesian_eef_step_, "safe_raise");
  // }
  reached_observation = reached_observation && move_arm_to_pose_joint(observation_pose);
  ok = ok && reached_observation;

  double object_world_yaw = 0.0;
  bool have_yaw = false;

  // Only estimate yaw after confirming that the camera has reached the intended
  // observation pose. If the move fails, keep using the fixed fallback grasp.
  if (reached_observation) {
    rclcpp::sleep_for(std::chrono::milliseconds(350));

    geometry_msgs::msg::PointStamped obj_for_yaw;
    obj_for_yaw.header.frame_id = src_frame;
    obj_for_yaw.header.stamp = rclcpp::Time(0);
    obj_for_yaw.point = request->object_point.point;

    PointCPtr cloud;
    std::string cloud_frame_id;
    if (capture_latest_cloud(cloud, cloud_frame_id, 2.0)) {
      if (shape == "nought") {
        std::vector<std::pair<double, double>> xy_pts;
        double ox = 0.0;
        double oy = 0.0;
        double ground_z = 0.0;
        if (collect_task1_object_top_points(
            tf_buffer_, node_->get_logger(), cloud, cloud_frame_id, obj_for_yaw, xy_pts, ox, oy, ground_z) &&
          estimate_nought_inner_edge_yaw_impl(node_->get_logger(), xy_pts, object_world_yaw))
        {
          have_yaw = true;
          RCLCPP_INFO(
            node_->get_logger(),
            "[Task1] estimated nought edge yaw = %.3f rad (%.1f deg), samples=%zu, ground_z=%.3f",
            object_world_yaw,
            object_world_yaw * 180.0 / M_PI,
            xy_pts.size(),
            ground_z);
        }
      } else {
        have_yaw = estimate_object_yaw_from_cloud(cloud, cloud_frame_id, obj_for_yaw, object_world_yaw);
      }
    } else {
      RCLCPP_WARN(node_->get_logger(), "[Task1] could not capture cloud for yaw estimate");
    }
  } else {
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task1] observation move failed; skip yaw estimation and use fixed fallback");
  }

  if (!have_yaw) {
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task1] falling back to fixed orientation for shape '%s'",
      shape.c_str());
  }

  // Apply the previously requested offset logic after yaw estimation. When a
  // principal yaw is available, rotate the local shape-specific offset into the
  // world frame before updating both the pick and place targets.
  double off_x = 0.0;
  double off_y = 0.0;
  if (task1_apply_shape_xy_offset_) {
    double local_x = 0.0;
    double local_y = 0.0;
    if (shape == "nought") {
      local_x = nought_grasp_offset_world_x_;
      local_y = nought_grasp_offset_world_y_;
    } else if (shape == "cross") {
      local_x = cross_grasp_offset_world_x_;
      local_y = cross_grasp_offset_world_y_;
    }

    if (have_yaw) {
      const double c = std::cos(object_world_yaw);
      const double s = std::sin(object_world_yaw);
      off_x = c * local_x - s * local_y;
      off_y = s * local_x + c * local_y;
    } else {
      off_x = local_x;
      off_y = local_y;
    }
  }

  if (task1_apply_shape_xy_offset_ && (off_x != 0.0 || off_y != 0.0)) {
    RCLCPP_INFO(
      node_->get_logger(),
      "Task1: using world XY offset (%.3f, %.3f) m for pick and place",
      off_x, off_y);
  }

  geometry_msgs::msg::Point obj_world = request->object_point.point;
  obj_world.x += off_x;
  obj_world.y += off_y;

  geometry_msgs::msg::Point goal_world = request->goal_point.point;
  goal_world.x += off_x;
  goal_world.y += off_y;

  geometry_msgs::msg::Point obj_p = transform_point_to_planning_frame(obj_world, src_frame);
  geometry_msgs::msg::Point goal_p = transform_point_to_planning_frame(goal_world, goal_src);

  geometry_msgs::msg::Quaternion grasp_q =
    have_yaw ? ee_orientation_for_shape_with_world_yaw(shape, object_world_yaw)
             : ee_orientation_for_shape(shape);

  geometry_msgs::msg::Pose pre_grasp;
  pre_grasp.position = obj_p;
  pre_grasp.position.z += pick_offset_z_;
  pre_grasp.orientation = grasp_q;

  geometry_msgs::msg::Pose grasp = pre_grasp;
  grasp.position.z -= grasp_descent_z_;

  geometry_msgs::msg::Pose lift = pre_grasp;
  lift.position.z = k_task1_safe_travel_z;

  geometry_msgs::msg::Pose pre_place;
  pre_place.position.x = goal_p.x;
  pre_place.position.y = goal_p.y;
  pre_place.position.z = k_task1_safe_travel_z;
  pre_place.orientation = grasp_q;

  geometry_msgs::msg::Pose place_release = pre_place;
  place_release.position.z = goal_p.z + place_offset_z_;

  const double grasp_w = gripper_grasp_width_;

  // Use joint-space planning for long travel moves and Cartesian motion only
  // for the short vertical approach, retreat, and placement descent.
  ok = ok && move_arm_to_pose_joint(pre_grasp);
  ok = ok && execute_cartesian_to_target(grasp, cartesian_eef_step_, "grasp");
  ok = ok && set_gripper_width(grasp_w);
  rclcpp::sleep_for(std::chrono::milliseconds(120));
  ok = ok && execute_cartesian_to_target(lift, cartesian_eef_step_, "lift");
  ok = ok && move_arm_to_pose_joint(pre_place);
  ok = ok && execute_cartesian_to_target(place_release, cartesian_eef_step_, "place_descent");
  ok = ok && set_gripper_width(gripper_open_width_);

  if (ok) {
    RCLCPP_INFO(node_->get_logger(), "Task 1 finished successfully");
  } else {
    RCLCPP_WARN(node_->get_logger(), "Task 1 finished with at least one failure");
  }
}

void cw2::t2_callback
(
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