#include <cw2_class.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
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

struct RoughClusterPoint
{
  double x;
  double y;
  double z;
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
};

struct RoughObject
{
  std::string kind;
  geometry_msgs::msg::Point center;
  double size_x;
  double size_y;
  double size_z;
  double top_z;
  double mean_r;
  double mean_g;
  double mean_b;
  std::size_t point_count;

  std::string rough_shape_label = "unknown";
  int cross_votes = 0;
  int nought_votes = 0;
  int unknown_votes = 0;
  int seen_views = 0;
};

struct RoughShapeCandidateCache
{
  geometry_msgs::msg::PointStamped point;
  std::string rough_label;
  int cross_votes;
  int nought_votes;
  int unknown_votes;
  int seen_views;
  double size_x;
  double size_y;
  double size_z;
  std::size_t point_count;
};

std::vector<RoughShapeCandidateCache> g_last_rough_candidates;

void refresh_rough_shape_label(RoughObject & obj)
{
  if (obj.cross_votes > obj.nought_votes && obj.cross_votes > obj.unknown_votes) {
    obj.rough_shape_label = "cross";
  } else if (obj.nought_votes > obj.cross_votes && obj.nought_votes > obj.unknown_votes) {
    obj.rough_shape_label = "nought";
  } else {
    obj.rough_shape_label = "unknown";
  }
}


// use top to estimate yaw and distinguish nought vs cross
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

  const std::size_t n = cloud->size();
  const std::size_t stride = std::max<std::size_t>(1, n / 50000);

  std::vector<float> ground_z_samples;
  ground_z_samples.reserve(n / stride + 1);

  constexpr double k_ground_sample_radius = 0.14;
  for (std::size_t i = 0; i < n; i += stride) {
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
  const std::size_t gi =
    std::min(static_cast<std::size_t>(ground_z_samples.size() * 0.06), ground_z_samples.size() - 1);
  ground_z = static_cast<double>(ground_z_samples[gi]);
  const float z_min = static_cast<float>(ground_z) + 0.012f;

  constexpr double k_object_crop_radius = 0.11;
  xy_pts.reserve(n / stride + 1);
  for (std::size_t i = 0; i < n; i += stride) {
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

// cross: pca
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

// nought: inner edge
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

/* ============================ constructor / utils ============================ */

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
    "cw2 refactored version initialised with pointcloud topic '%s' (%s QoS)",
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

// top down
geometry_msgs::msg::Quaternion cw2::make_top_down_q() const
{
  tf2::Quaternion q;
  q.setRPY(3.14159265358979323846, 0.0, -0.78539816339744830962);
  q.normalize();
  geometry_msgs::msg::Quaternion msg;
  tf2::convert(q, msg);
  return msg;
}

// top down + estimated yaw fallback
geometry_msgs::msg::Quaternion cw2::ee_orientation_for_shape(const std::string & shape) const
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

// top down + estimated yaw
geometry_msgs::msg::Quaternion cw2::ee_orientation_for_shape_with_world_yaw(
  const std::string & shape,
  double world_yaw_rad) const
{
  const double base_yaw = -0.78539816339744830962;
  double final_yaw = base_yaw;

  if (shape == "cross") {
    final_yaw += world_yaw_rad;
  } else if (shape == "nought") {
    final_yaw += world_yaw_rad + nought_grasp_extra_yaw_rad_;
  }

  tf2::Quaternion q;
  q.setRPY(3.14159265358979323846, 0.0, final_yaw);
  q.normalize();

  geometry_msgs::msg::Quaternion msg;
  tf2::convert(q, msg);
  return msg;
}

// moveit
geometry_msgs::msg::Point cw2::transform_point_to_planning_frame(
  const geometry_msgs::msg::Point & p,
  const std::string & source_frame)
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
        "TF %s -> %s failed: %s", frame.c_str(), planning_frame.c_str(), ex.what());
    }
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "TF failed for point transform from '%s'; using untransformed point",
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

// estimate cross yaw
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
  const PointCPtr & cloud,
  const std::string & cloud_frame_id,
  const geometry_msgs::msg::PointStamped & object_loc)
{
  if (!cloud || cloud->empty() || cloud_frame_id.empty()) {
    RCLCPP_WARN(node_->get_logger(), "[Task2] classify: no cloud/frame");
    return "unknown";
  }

  geometry_msgs::msg::TransformStamped tf_geom;
  try {
    tf_geom = tf_buffer_.lookupTransform(
      "world", cloud_frame_id, rclcpp::Time(0), tf2::durationFromSec(2.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(node_->get_logger(), "[Task2] TF world<-cloud failed: %s", ex.what());
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
  } catch (const tf2::TransformException &) {
  }

  const std::size_t n = cloud->size();
  const std::size_t stride = std::max<std::size_t>(1, n / 50000);

  std::vector<float> ground_z_samples;
  ground_z_samples.reserve(n / stride + 1);
  for (std::size_t i = 0; i < n; i += stride) {
    const auto & pt = cloud->points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    const tf2::Vector3 vc(pt.x, pt.y, pt.z);
    const tf2::Vector3 vw = T_w_c * vc;
    const double dh = std::hypot(vw.x() - ox, vw.y() - oy);
    if (dh < task2_ground_sample_radius_m_) {
      ground_z_samples.push_back(static_cast<float>(vw.z()));
    }
  }
  if (ground_z_samples.empty()) {
    return "unknown";
  }
  std::sort(ground_z_samples.begin(), ground_z_samples.end());
  const std::size_t gi =
    std::min(static_cast<std::size_t>(ground_z_samples.size() * 0.06), ground_z_samples.size() - 1);
  const float ground_z = ground_z_samples[gi];
  const float z_obj_min = ground_z + static_cast<float>(task2_surface_above_ground_m_);

  std::int64_t inner = 0;
  std::int64_t ring = 0;
  for (std::size_t i = 0; i < n; i += stride) {
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

  const double ratio = static_cast<double>(ring) / static_cast<double>(inner + 8);

    std::string label = "unknown";

    if (inner >= task2_inner_min_points_cross_) {
      label = "cross";
    } else if (
      ring >= 120 &&
      inner <= 20 &&
      ratio >= task2_ring_inner_ratio_nought_min_)
    {
      label = "nought";
    }

  RCLCPP_INFO(
    node_->get_logger(),
    "[Task2] classified '%s' at (%.3f, %.3f): inner=%lld ring=%lld ratio=%.2f",
    label.c_str(), ox, oy,
    static_cast<long long>(inner),
    static_cast<long long>(ring),
    ratio);

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
  const geometry_msgs::msg::Pose & target_pose,
  double eef_step,
  const char * step_name)
{
  const double safe_eef_step = std::clamp(eef_step, 0.005, 0.02);
  const double jump_threshold = 0.0;
  const char * label = (step_name && step_name[0]) ? step_name : "move";

  arm_group_->setStartStateToCurrentState();
  arm_group_->setMaxVelocityScalingFactor(0.22);
  arm_group_->setMaxAccelerationScalingFactor(0.22);

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory traj;
  const double fraction =
    arm_group_->computeCartesianPath(waypoints, safe_eef_step, jump_threshold, traj);

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

/* ============================ extracted functions ============================ */

// task 2 + 3
std::string cw2::deep_scan(const geometry_msgs::msg::PointStamped & object_loc)
{
  if (!move_to_task2_observation_pose(object_loc)) {
    RCLCPP_WARN(
      node_->get_logger(),
      "[DeepScan] could not reach observation pose for (%.3f, %.3f)",
      object_loc.point.x, object_loc.point.y);
    return "unknown";
  }

  rclcpp::sleep_for(std::chrono::milliseconds(static_cast<int>(task2_settle_ms_)));

  PointCPtr cloud;
  std::string frame_id;
  if (!capture_latest_cloud(cloud, frame_id, 2.0)) {
    RCLCPP_WARN(node_->get_logger(), "[DeepScan] no cloud available");
    return "unknown";
  }

  return classify_shape_from_cloud(cloud, frame_id, object_loc);
}

// task 1 + 3
bool cw2::pick_and_place(
  const geometry_msgs::msg::PointStamped & object_point,
  const geometry_msgs::msg::PointStamped & goal_point,
  const std::string & shape_type)
{
  const std::string src_frame =
    object_point.header.frame_id.empty() ? "world" : object_point.header.frame_id;
  const std::string goal_frame =
    goal_point.header.frame_id.empty() ? "world" : goal_point.header.frame_id;

  const geometry_msgs::msg::Point nominal_obj_p =
    transform_point_to_planning_frame(object_point.point, src_frame);

  const double k_safe_travel_z = 0.60;
  const geometry_msgs::msg::Quaternion top_down_q = make_top_down_q();

  geometry_msgs::msg::Pose observation_pose;
  observation_pose.position.x = nominal_obj_p.x;
  observation_pose.position.y = nominal_obj_p.y;
  observation_pose.position.z = k_safe_travel_z;
  observation_pose.orientation = top_down_q;

  bool ok = true;
  ok = ok && set_gripper_width(gripper_open_width_);

  const bool reached_observation = move_arm_to_pose_joint(observation_pose);
  ok = ok && reached_observation;

  double object_world_yaw = 0.0;
  bool have_yaw = false;

  if (reached_observation) {
    rclcpp::sleep_for(std::chrono::milliseconds(350));

    PointCPtr cloud;
    std::string cloud_frame_id;
    if (capture_latest_cloud(cloud, cloud_frame_id, 2.0)) {
      geometry_msgs::msg::PointStamped obj_for_yaw = object_point;

      if (shape_type == "nought") {
        std::vector<std::pair<double, double>> xy_pts;
        double ox = 0.0;
        double oy = 0.0;
        double ground_z = 0.0;

        if (collect_task1_object_top_points(
              tf_buffer_, node_->get_logger(), cloud, cloud_frame_id,
              obj_for_yaw, xy_pts, ox, oy, ground_z) &&
            estimate_nought_inner_edge_yaw_impl(
              node_->get_logger(), xy_pts, object_world_yaw))
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
        have_yaw = estimate_object_yaw_from_cloud(
          cloud, cloud_frame_id, obj_for_yaw, object_world_yaw);
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
      shape_type.c_str());
  }

  double off_x = 0.0;
  double off_y = 0.0;
  if (task1_apply_shape_xy_offset_) {
    double local_x = 0.0;
    double local_y = 0.0;

    if (shape_type == "nought") {
      local_x = nought_grasp_offset_world_x_;
      local_y = nought_grasp_offset_world_y_;
    } else if (shape_type == "cross") {
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

  geometry_msgs::msg::Point obj_world = object_point.point;
  obj_world.x += off_x;
  obj_world.y += off_y;

  geometry_msgs::msg::Point goal_world = goal_point.point;
  goal_world.x += off_x;
  goal_world.y += off_y;

  const geometry_msgs::msg::Point obj_p =
    transform_point_to_planning_frame(obj_world, src_frame);
  const geometry_msgs::msg::Point goal_p =
    transform_point_to_planning_frame(goal_world, goal_frame);

  const geometry_msgs::msg::Quaternion grasp_q =
    have_yaw ? ee_orientation_for_shape_with_world_yaw(shape_type, object_world_yaw)
             : ee_orientation_for_shape(shape_type);

  geometry_msgs::msg::Pose pre_grasp;
  pre_grasp.position = obj_p;
  pre_grasp.position.z += pick_offset_z_;
  pre_grasp.orientation = grasp_q;

  geometry_msgs::msg::Pose grasp = pre_grasp;
  grasp.position.z -= grasp_descent_z_;

  geometry_msgs::msg::Pose lift = pre_grasp;
  lift.position.z = k_safe_travel_z;

  geometry_msgs::msg::Pose pre_place;
  pre_place.position.x = goal_p.x;
  pre_place.position.y = goal_p.y;
  pre_place.position.z = k_safe_travel_z;
  pre_place.orientation = grasp_q;

  geometry_msgs::msg::Pose place_release = pre_place;
  place_release.position.z = goal_p.z + place_offset_z_;

  ok = ok && move_arm_to_pose_joint(pre_grasp);
  ok = ok && move_arm_linear_to(grasp, cartesian_eef_step_, "grasp");
  ok = ok && set_gripper_width(gripper_grasp_width_);
  rclcpp::sleep_for(std::chrono::milliseconds(120));
  ok = ok && move_arm_linear_to(lift, cartesian_eef_step_, "lift");
  ok = ok && move_arm_to_pose_joint(pre_place);
  ok = ok && move_arm_linear_to(place_release, cartesian_eef_step_, "place_descent");
  ok = ok && set_gripper_width(gripper_open_width_);

  if (ok) {
    RCLCPP_INFO(node_->get_logger(), "[PickPlace] finished successfully");
  } else {
    RCLCPP_WARN(node_->get_logger(), "[PickPlace] finished with at least one failure");
  }

  return ok;
}

// task 3
std::vector<geometry_msgs::msg::PointStamped> cw2::rough_scan(
  geometry_msgs::msg::PointStamped & basket_point)
{
  g_last_rough_candidates.clear();

  auto merge_rough_object =
    [](std::vector<RoughObject> & merged, const RoughObject & fresh)
    {
      double merge_xy = 0.07;
      if (fresh.kind == "basket") {
        merge_xy = 0.14;
      } else if (fresh.kind == "obstacle") {
        merge_xy = 0.08;
      }

      for (auto & m : merged) {
        if (m.kind != fresh.kind) {
          continue;
        }

        const double dxy = std::hypot(m.center.x - fresh.center.x, m.center.y - fresh.center.y);
        if (dxy < merge_xy) {
          m.center.x = 0.5 * (m.center.x + fresh.center.x);
          m.center.y = 0.5 * (m.center.y + fresh.center.y);
          m.center.z = 0.5 * (m.center.z + fresh.center.z);
          m.size_x = std::max(m.size_x, fresh.size_x);
          m.size_y = std::max(m.size_y, fresh.size_y);
          m.size_z = std::max(m.size_z, fresh.size_z);
          m.top_z = std::max(m.top_z, fresh.top_z);
          m.mean_r = 0.5 * (m.mean_r + fresh.mean_r);
          m.mean_g = 0.5 * (m.mean_g + fresh.mean_g);
          m.mean_b = 0.5 * (m.mean_b + fresh.mean_b);
          m.point_count = std::max(m.point_count, fresh.point_count);

          m.cross_votes += fresh.cross_votes;
          m.nought_votes += fresh.nought_votes;
          m.unknown_votes += fresh.unknown_votes;
          m.seen_views += fresh.seen_views;
          refresh_rough_shape_label(m);
          return;
        }
      }

      merged.push_back(fresh);
    };

  auto extract_rough_objects =
    [this](const PointCPtr & cloud, const std::string & frame_id) -> std::vector<RoughObject>
    {
      std::vector<RoughObject> objects;
      if (!cloud || cloud->empty() || frame_id.empty()) {
        return objects;
      }

      geometry_msgs::msg::TransformStamped tf_geom;
      try {
        tf_geom = tf_buffer_.lookupTransform(
          "world", frame_id, rclcpp::Time(0), tf2::durationFromSec(2.0));
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(node_->get_logger(), "[RoughScan] TF world<-cloud failed: %s", ex.what());
        return objects;
      }

      tf2::Transform T_w_c;
      tf2::fromMsg(tf_geom.transform, T_w_c);

      const std::size_t n = cloud->size();
      const std::size_t stride = std::max<std::size_t>(1, n / 25000);

      std::vector<double> all_z;
      all_z.reserve(n / stride + 1);
      std::vector<RoughClusterPoint> roi_pts;
      roi_pts.reserve(n / stride + 1);

      for (std::size_t i = 0; i < n; i += stride) {
        const auto & pt = cloud->points[i];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
          continue;
        }

        const tf2::Vector3 vc(pt.x, pt.y, pt.z);
        const tf2::Vector3 vw = T_w_c * vc;
        if (vw.x() < -0.75 || vw.x() > 0.85 || vw.y() < -0.65 || vw.y() > 0.65) {
          continue;
        }

        all_z.push_back(vw.z());
        roi_pts.push_back({vw.x(), vw.y(), vw.z(), pt.r, pt.g, pt.b});
      }

      RCLCPP_INFO(
        node_->get_logger(),
        "[RoughScan] roi_pts=%zu all_z=%zu",
        roi_pts.size(), all_z.size());

      if (roi_pts.size() < 50 || all_z.size() < 50) {
        return objects;
      }

      std::sort(all_z.begin(), all_z.end());
      const std::size_t gi =
        std::min(all_z.size() - 1, static_cast<std::size_t>(0.08 * all_z.size()));
      const double ground_z = all_z[gi];

      std::vector<RoughClusterPoint> elevated;
      elevated.reserve(roi_pts.size());
      for (const auto & p : roi_pts) {
        const double dz = p.z - ground_z;

        const bool strongly_green =
          (static_cast<int>(p.g) > static_cast<int>(p.r) + 20) &&
          (static_cast<int>(p.g) > static_cast<int>(p.b) + 20) &&
          (p.g > 70);

        // 桌面：又绿、又贴地，直接排除
        if (strongly_green && dz < 0.03) {
          continue;
        }

        if (dz >= 0.003 && dz <= 0.18) {
          elevated.push_back(p);
        }
      }

      RCLCPP_INFO(
        node_->get_logger(),
        "[RoughScan] ground_z=%.3f elevated=%zu",
        ground_z, elevated.size());

      if (elevated.size() < 20) {
        return objects;
      }

      std::vector<char> used(elevated.size(), 0);
      constexpr double k_cluster_r = 0.055;
      constexpr double k_cluster_r_sq = k_cluster_r * k_cluster_r;
      constexpr double k_cluster_dz = 0.07;

      for (std::size_t i = 0; i < elevated.size(); ++i) {
        if (used[i]) {
          continue;
        }

        std::vector<std::size_t> queue;
        queue.push_back(i);
        used[i] = 1;

        std::vector<std::size_t> idxs;
        for (std::size_t q = 0; q < queue.size(); ++q) {
          const std::size_t cur = queue[q];
          idxs.push_back(cur);

          for (std::size_t j = 0; j < elevated.size(); ++j) {
            if (used[j]) {
              continue;
            }

            const double dx = elevated[j].x - elevated[cur].x;
            const double dy = elevated[j].y - elevated[cur].y;
            const double dz = std::fabs(elevated[j].z - elevated[cur].z);
            if ((dx * dx + dy * dy) <= k_cluster_r_sq && dz <= k_cluster_dz) {
              used[j] = 1;
              queue.push_back(j);
            }
          }
        }

        RCLCPP_INFO(
          node_->get_logger(),
          "[RoughScan] raw cluster size=%zu",
          idxs.size());

        if (idxs.size() < 8) {
          continue;
        }

        double min_x = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        double min_z = std::numeric_limits<double>::infinity();
        double max_z = -std::numeric_limits<double>::infinity();
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        double sum_r = 0.0;
        double sum_g = 0.0;
        double sum_b = 0.0;

        for (const std::size_t idx : idxs) {
          const auto & p = elevated[idx];
          min_x = std::min(min_x, p.x);
          max_x = std::max(max_x, p.x);
          min_y = std::min(min_y, p.y);
          max_y = std::max(max_y, p.y);
          min_z = std::min(min_z, p.z);
          max_z = std::max(max_z, p.z);
          sum_x += p.x;
          sum_y += p.y;
          sum_z += p.z;
          sum_r += static_cast<double>(p.r);
          sum_g += static_cast<double>(p.g);
          sum_b += static_cast<double>(p.b);
        }

        RoughObject obj;
        const double inv_n = 1.0 / static_cast<double>(idxs.size());
        obj.center.x = sum_x * inv_n;
        obj.center.y = sum_y * inv_n;
        obj.center.z = sum_z * inv_n;
        obj.size_x = max_x - min_x;
        obj.size_y = max_y - min_y;
        obj.size_z = max_z - min_z;
        obj.top_z = max_z;
        obj.mean_r = sum_r * inv_n;
        obj.mean_g = sum_g * inv_n;
        obj.mean_b = sum_b * inv_n;
        obj.point_count = idxs.size();
        obj.kind = "unknown";
        obj.rough_shape_label = "unknown";
        obj.cross_votes = 0;
        obj.nought_votes = 0;
        obj.unknown_votes = 0;
        obj.seen_views = 0;

        RCLCPP_INFO(
          node_->get_logger(),
          "[RoughScan] cluster center=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f) points=%zu rgb=(%.1f, %.1f, %.1f)",
          obj.center.x, obj.center.y, obj.center.z,
          obj.size_x, obj.size_y, obj.size_z,
          obj.point_count,
          obj.mean_r, obj.mean_g, obj.mean_b);

        const double max_xy = std::max(obj.size_x, obj.size_y);
        const double mean_intensity = (obj.mean_r + obj.mean_g + obj.mean_b) / 3.0;
        const bool reddish =
          (obj.mean_r > obj.mean_g + 18.0) && (obj.mean_r > obj.mean_b + 18.0);
        const bool darkish =
          mean_intensity < 65.0 &&
          std::fabs(obj.mean_r - obj.mean_g) < 25.0 &&
          std::fabs(obj.mean_r - obj.mean_b) < 25.0;

          if (obj.size_x > 0.28 || obj.size_y > 0.28) {
            RCLCPP_INFO(
              node_->get_logger(),
              "[RoughScan] skip oversized cluster center=(%.3f, %.3f) size=(%.3f, %.3f, %.3f)",
              obj.center.x, obj.center.y, obj.size_x, obj.size_y, obj.size_z);
            continue;
          }

        if (max_xy > 0.22 && max_xy < 0.50 && obj.size_z < 0.10 && reddish) {
          obj.kind = "basket";
        } else if (max_xy > 0.03 && max_xy < 0.14 && obj.size_z > 0.03 && darkish) {
          obj.kind = "obstacle";
        } else if (max_xy > 0.04 && max_xy < 0.25 && obj.size_z > 0.006 && obj.size_z < 0.10) {
          obj.kind = "shape";
        } else {
          continue;
        }

        if (obj.kind == "shape") {
          int inner = 0;
          int ring = 0;

          for (const std::size_t idx : idxs) {
            const auto & p = elevated[idx];
            const double dx = p.x - obj.center.x;
            const double dy = p.y - obj.center.y;
            const double r = std::hypot(dx, dy);

            if (r < 0.022) {
              ++inner;
            } else if (r >= 0.028 && r <= 0.055) {
              ++ring;
            }
          }

          const double ratio = static_cast<double>(ring) / static_cast<double>(inner + 8);
          obj.seen_views = 1;

          if (inner >= 20) {
            obj.cross_votes = 1;
            obj.rough_shape_label = "cross";
          } else if (ratio >= 1.2) {
            obj.nought_votes = 1;
            obj.rough_shape_label = "nought";
          } else {
            obj.unknown_votes = 1;
            obj.rough_shape_label = "unknown";
          }

          RCLCPP_INFO(
            node_->get_logger(),
            "[RoughScan] rough shape @ (%.3f, %.3f): label=%s inner=%d ring=%d ratio=%.2f",
            obj.center.x, obj.center.y, obj.rough_shape_label.c_str(), inner, ring, ratio);
        }

        objects.push_back(obj);
      }

      return objects;
    };

  auto make_pose =
    [this](double x, double y, double z) -> geometry_msgs::msg::Pose
    {
      geometry_msgs::msg::Pose pose;
      pose.position.x = x;
      pose.position.y = y;
      pose.position.z = z;
      pose.orientation = make_top_down_q();
      return pose;
    };

  auto move_to_high_transit =
    [this, &make_pose]() -> bool
    {
      return move_arm_to_pose_joint(make_pose(0.30, -0.10, 0.55));
    };

  basket_point.header.frame_id = "";
  basket_point.header.stamp = rclcpp::Time(0);
  basket_point.point.x = 0.0;
  basket_point.point.y = 0.0;
  basket_point.point.z = 0.0;

  std::vector<RoughObject> merged;
  std::vector<geometry_msgs::msg::PointStamped> candidates;

  const std::vector<std::pair<double, double>> scan_xy = {
    {0.50, -0.35}, {0.60, 0.00}, {0.55, 0.40}, {0.00, 0.45},
    {0.00, -0.45}, {-0.55, -0.40}, {-0.60, 0.00}, {-0.50, 0.35}
  };

  RCLCPP_INFO(node_->get_logger(), "[RoughScan] starting multi-view scan");

  move_to_high_transit();
  set_gripper_width(gripper_open_width_);

  for (std::size_t i = 0; i < scan_xy.size(); ++i) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[RoughScan] scan pose %zu -> (%.3f, %.3f, 0.60)",
      i, scan_xy[i].first, scan_xy[i].second);

    bool ok = move_arm_to_pose_joint(make_pose(scan_xy[i].first, scan_xy[i].second, 0.60));

    if (!ok) {
      RCLCPP_WARN(node_->get_logger(), "[RoughScan] failed to reach scan pose %zu", i);
      continue;
    }

    rclcpp::sleep_for(std::chrono::milliseconds(300));

    for (int f = 0; f < 2; ++f) {
      PointCPtr cloud;
      std::string frame_id;
      if (!capture_latest_cloud(cloud, frame_id, 1.5)) {
        continue;
      }

      const auto objs = extract_rough_objects(cloud, frame_id);
      for (const auto & obj : objs) {
        merge_rough_object(merged, obj);
      }
      rclcpp::sleep_for(std::chrono::milliseconds(120));
    }
  }

  RoughObject best_basket;
  bool have_basket = false;
  for (const auto & obj : merged) {
    if (obj.kind == "basket") {
      if (!have_basket || obj.point_count > best_basket.point_count) {
        best_basket = obj;
        have_basket = true;
      }
    }
  }

  if (have_basket) {
    basket_point.header.frame_id = "world";
    basket_point.header.stamp = rclcpp::Time(0);
    basket_point.point = best_basket.center;
    basket_point.point.z = best_basket.top_z;
    RCLCPP_INFO(
      node_->get_logger(),
      "[RoughScan] basket detected at (%.3f, %.3f, %.3f)",
      basket_point.point.x, basket_point.point.y, basket_point.point.z);
  } else {
    RCLCPP_WARN(node_->get_logger(), "[RoughScan] basket not detected");
  }

for (auto & obj : merged) {
  if (obj.kind != "shape") {
    continue;
  }

  // 小簇噪声
  if (obj.point_count < 80) {
    continue;
  }
  // 几何过滤：去掉明显不合理的假候选
  if (obj.size_x < 0.03 || obj.size_y < 0.03) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[RoughScan] reject tiny candidate at (%.3f, %.3f) size=(%.3f, %.3f, %.3f)",
      obj.center.x, obj.center.y, obj.size_x, obj.size_y, obj.size_z);
    continue;
  }

  if (obj.size_x > 0.26 || obj.size_y > 0.26) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[RoughScan] reject oversized shape candidate at (%.3f, %.3f) size=(%.3f, %.3f, %.3f)",
      obj.center.x, obj.center.y, obj.size_x, obj.size_y, obj.size_z);
    continue;
  }

  const double aspect =
    std::max(obj.size_x, obj.size_y) / std::max(1e-6, std::min(obj.size_x, obj.size_y));
  if (aspect > 6.0) {
    RCLCPP_INFO(
      node_->get_logger(),
      "[RoughScan] reject elongated candidate at (%.3f, %.3f) size=(%.3f, %.3f, %.3f) aspect=%.2f",
      obj.center.x, obj.center.y, obj.size_x, obj.size_y, obj.size_z, aspect);
    continue;
  }

  refresh_rough_shape_label(obj);

  geometry_msgs::msg::PointStamped p;
  p.header.frame_id = "world";
  p.header.stamp = rclcpp::Time(0);
  p.point = obj.center;
  candidates.push_back(p);

  RoughShapeCandidateCache cache;
  cache.point = p;
  cache.rough_label = obj.rough_shape_label;
  cache.cross_votes = obj.cross_votes;
  cache.nought_votes = obj.nought_votes;
  cache.unknown_votes = obj.unknown_votes;
  cache.seen_views = obj.seen_views;
  cache.size_x = obj.size_x;
  cache.size_y = obj.size_y;
  cache.size_z = obj.size_z;
  cache.point_count = obj.point_count;
  g_last_rough_candidates.push_back(cache);

  RCLCPP_INFO(
    node_->get_logger(),
    "[RoughScan] candidate shape at (%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f) rough=%s votes(c=%d,n=%d,u=%d) seen=%d",
    p.point.x, p.point.y, p.point.z,
    obj.size_x, obj.size_y, obj.size_z,
    obj.rough_shape_label.c_str(),
    obj.cross_votes, obj.nought_votes, obj.unknown_votes, obj.seen_views);
}

  move_to_high_transit();
  return candidates;
}

/* ============================ callbacks ============================ */

void cw2::t1_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response)
{
  (void)response;

  RCLCPP_INFO(
    node_->get_logger(),
    "Task1 request: shape=%s object=(%.3f, %.3f, %.3f) goal=(%.3f, %.3f, %.3f)",
    request->shape_type.c_str(),
    request->object_point.point.x, request->object_point.point.y, request->object_point.point.z,
    request->goal_point.point.x, request->goal_point.point.y, request->goal_point.point.z);

  pick_and_place(request->object_point, request->goal_point, request->shape_type);
}

void cw2::t2_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response)
{
  response->mystery_object_num = -1;

  if (request->ref_object_points.size() < 2) {
    RCLCPP_ERROR(node_->get_logger(), "[Task2] need two reference points");
    return;
  }

  const std::string s0 = deep_scan(request->ref_object_points[0]);
  const std::string s1 = deep_scan(request->ref_object_points[1]);
  const std::string sm = deep_scan(request->mystery_object_point);

  if (sm != "unknown") {
    if (sm == s0 && sm != s1) {
      response->mystery_object_num = 1;
    } else if (sm == s1 && sm != s0) {
      response->mystery_object_num = 2;
    } else if (sm == s0 && sm == s1) {
      response->mystery_object_num = 1;
    }
  }

  RCLCPP_INFO(
    node_->get_logger(),
    "[Task2] ref1=%s ref2=%s mystery=%s answer=%lld",
    s0.c_str(), s1.c_str(), sm.c_str(),
    static_cast<long long>(response->mystery_object_num));
}

void cw2::t3_callback(
  const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
  std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response)
{
  (void)request;
  response->total_num_shapes = 0;
  response->num_most_common_shape = 0;
  response->most_common_shape_vector.clear();

  geometry_msgs::msg::PointStamped basket_point;
  const auto candidates = rough_scan(basket_point);

  if (candidates.empty()) {
    RCLCPP_WARN(node_->get_logger(), "[Task3] no candidate shapes found");
    return;
  }

  auto find_rough_info =
    [](const geometry_msgs::msg::PointStamped & p) -> const RoughShapeCandidateCache * {
      const RoughShapeCandidateCache * best = nullptr;
      double best_d = std::numeric_limits<double>::infinity();

      for (const auto & info : g_last_rough_candidates) {
        const double d = std::hypot(
          info.point.point.x - p.point.x,
          info.point.point.y - p.point.y);
        if (d < best_d) {
          best_d = d;
          best = &info;
        }
      }

      if (best && best_d < 0.10) {
        return best;
      }
      return nullptr;
    };

  auto high_confidence_rough =
    [](const RoughShapeCandidateCache & info) -> bool {
      if (info.rough_label == "cross") {
        return info.seen_views >= 2 && info.cross_votes >= info.nought_votes + 1;
      }
      if (info.rough_label == "nought") {
        return info.seen_views >= 2 && info.nought_votes >= info.cross_votes + 1;
      }
      return false;
    };

  std::vector<std::string> labels;
  labels.reserve(candidates.size());

  int n_cross = 0;
  int n_nought = 0;
  int deep_scans_used = 0;

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    std::string label = "unknown";

    const RoughShapeCandidateCache * rough = find_rough_info(candidates[i]);
    if (rough && high_confidence_rough(*rough)) {
      label = rough->rough_label;
      RCLCPP_INFO(
        node_->get_logger(),
        "[Task3] candidate %zu at (%.3f, %.3f) accepted rough=%s votes(c=%d,n=%d,u=%d) seen=%d",
        i,
        candidates[i].point.x, candidates[i].point.y,
        label.c_str(),
        rough->cross_votes, rough->nought_votes, rough->unknown_votes, rough->seen_views);
    } else {
      label = deep_scan(candidates[i]);
      ++deep_scans_used;

      if (rough) {
        RCLCPP_INFO(
          node_->get_logger(),
          "[Task3] candidate %zu at (%.3f, %.3f) rough=%s uncertain -> deep=%s",
          i,
          candidates[i].point.x, candidates[i].point.y,
          rough->rough_label.c_str(),
          label.c_str());
      } else {
        RCLCPP_INFO(
          node_->get_logger(),
          "[Task3] candidate %zu at (%.3f, %.3f) no rough match -> deep=%s",
          i,
          candidates[i].point.x, candidates[i].point.y,
          label.c_str());
      }
    }

    labels.push_back(label);

    if (label == "cross") {
      ++n_cross;
    } else if (label == "nought") {
      ++n_nought;
    }
  }

  response->total_num_shapes = static_cast<std::int64_t>(n_cross + n_nought);
  response->num_most_common_shape =
    static_cast<std::int64_t>(std::max(n_cross, n_nought));

  RCLCPP_INFO(
    node_->get_logger(),
    "[Task3] deep scans used = %d / %zu",
    deep_scans_used, candidates.size());

  if (response->total_num_shapes <= 0) {
    RCLCPP_WARN(node_->get_logger(), "[Task3] no valid shapes after rough/deep classification");
    return;
  }

  if (basket_point.header.frame_id.empty()) {
    RCLCPP_WARN(node_->get_logger(), "[Task3] basket missing; counting only");
    return;
  }

  const std::string target_shape = (n_cross >= n_nought) ? "cross" : "nought";

  int best_idx = -1;
  double best_score = -1e9;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (labels[i] != target_shape) {
      continue;
    }

    const double score =
      1.5 * std::hypot(candidates[i].point.x, candidates[i].point.y) -
      0.7 * std::abs(candidates[i].point.y);

    if (score > best_score) {
      best_score = score;
      best_idx = static_cast<int>(i);
    }
  }

  if (best_idx < 0) {
    RCLCPP_WARN(node_->get_logger(), "[Task3] no target of type '%s' found", target_shape.c_str());
    return;
  }

  geometry_msgs::msg::PointStamped place_point = basket_point;
  place_point.point.x += (basket_point.point.x >= 0.0) ? -0.02 : 0.02;
  place_point.point.y += (basket_point.point.y >= 0.0) ? -0.02 : 0.02;

  RCLCPP_INFO(
    node_->get_logger(),
    "[Task3] total=%lld cross=%d nought=%d target_shape=%s target_idx=%d",
    static_cast<long long>(response->total_num_shapes),
    n_cross, n_nought, target_shape.c_str(), best_idx);

  pick_and_place(candidates[static_cast<std::size_t>(best_idx)], place_point, target_shape);
}