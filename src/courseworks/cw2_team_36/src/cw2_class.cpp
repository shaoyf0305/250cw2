#include <cw2_class.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/io/pcd_io.h>
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

struct Task3MergedScanObject
{
  std::string category;
  geometry_msgs::msg::Point position;
  double top_z = 0.0;
  double size_x = 0.0;
  double size_y = 0.0;
  double size_z = 0.0;
  double mean_r = 0.0;
  double mean_g = 0.0;
  double mean_b = 0.0;
  std::size_t point_count = 0;
  int seen_count = 0;
  int cross_votes = 0;
  int nought_votes = 0;
  int unknown_votes = 0;
  std::vector<pcl::PointXYZRGB> vis_points;

  int best_pose_idx = -1;
  double best_view_distance_xy = std::numeric_limits<double>::infinity();
  std::string best_view_category = "unknown";
  geometry_msgs::msg::Point best_view_position;
  double best_view_size_x = 0.0;
  double best_view_size_y = 0.0;
  double best_view_size_z = 0.0;
};

std::vector<Task3MergedScanObject> g_last_task3_objects;

constexpr double k_basket_pad_x = 0.020;
constexpr double k_basket_pad_y = 0.020;
constexpr double k_basket_z_min_offset = -0.006;
constexpr double k_basket_z_max_offset = 0.110;

bool is_warm_red_point(const pcl::PointXYZRGB & p)
{
  return static_cast<double>(p.r) > 40.0 &&
         static_cast<double>(p.r) > static_cast<double>(p.g) + 12.0 &&
         static_cast<double>(p.r) > static_cast<double>(p.b) + 12.0;
}

bool is_dark_neutral_point(const pcl::PointXYZRGB & p)
{
  const double max_rgb = std::max(
    {static_cast<double>(p.r), static_cast<double>(p.g), static_cast<double>(p.b)});
  const double min_rgb = std::min(
    {static_cast<double>(p.r), static_cast<double>(p.g), static_cast<double>(p.b)});
  return max_rgb < 36.0 && (max_rgb - min_rgb) < 14.0;
}

bool looks_like_rim_only_basket(
  double size_x,
  double size_y,
  double size_z,
  double mean_r,
  double mean_g,
  double mean_b,
  double center_fill_ratio,
  double ring_fill_ratio)
{
  const double long_side = std::max(size_x, size_y);
  const double short_side = std::min(size_x, size_y);

  const bool basket_warm =
    mean_r > 45.0 && mean_g > 18.0 && mean_b > 18.0 &&
    mean_r > mean_g + 20.0 && mean_r > mean_b + 20.0;

  const bool basket_low = size_z > 0.010 && size_z < 0.12;
  const bool rim_span_ok = long_side > 0.33 && short_side > 0.24;
  const bool hollow_like = center_fill_ratio < 0.10 && ring_fill_ratio > 0.12;

  return basket_warm && basket_low && rim_span_ok && hollow_like;
}

void append_local_roi_points(
  const std::vector<pcl::PointXYZRGB> & roi_pts,
  const geometry_msgs::msg::Point & center,
  double size_x,
  double size_y,
  double z_min,
  double z_max,
  double pad_x,
  double pad_y,
  bool warm_only,
  bool dark_only,
  std::vector<pcl::PointXYZRGB> & out)
{
  const double hx = 0.5 * size_x + pad_x;
  const double hy = 0.5 * size_y + pad_y;

  for (const auto & p : roi_pts) {
    if (static_cast<double>(p.x) < center.x - hx || static_cast<double>(p.x) > center.x + hx ||
        static_cast<double>(p.y) < center.y - hy || static_cast<double>(p.y) > center.y + hy) {
      continue;
    }
    if (static_cast<double>(p.z) < z_min || static_cast<double>(p.z) > z_max) {
      continue;
    }
    if (warm_only && !is_warm_red_point(p)) {
      continue;
    }
    if (dark_only && !is_dark_neutral_point(p)) {
      continue;
    }
    out.push_back(p);
  }
}

void compute_hollow_metrics(
  const std::vector<pcl::PointXYZRGB> & cluster_pts,
  const geometry_msgs::msg::Point & center,
  double & center_fill_ratio,
  double & ring_fill_ratio,
  double & max_r_metric)
{
  center_fill_ratio = 1.0;
  ring_fill_ratio = 0.0;
  max_r_metric = 0.0;
  if (cluster_pts.empty()) {
    return;
  }

  for (const auto & p : cluster_pts) {
    const double dx = static_cast<double>(p.x) - center.x;
    const double dy = static_cast<double>(p.y) - center.y;
    max_r_metric = std::max(max_r_metric, std::hypot(dx, dy));
  }
  if (max_r_metric < 1e-6) {
    return;
  }

  int n_center = 0;
  int n_ring = 0;
  for (const auto & p : cluster_pts) {
    const double dx = static_cast<double>(p.x) - center.x;
    const double dy = static_cast<double>(p.y) - center.y;
    const double r = std::hypot(dx, dy);
    if (r < 0.22 * max_r_metric) {
      ++n_center;
    }
    if (r >= 0.45 * max_r_metric && r <= 0.95 * max_r_metric) {
      ++n_ring;
    }
  }

  center_fill_ratio = static_cast<double>(n_center) / static_cast<double>(cluster_pts.size());
  ring_fill_ratio = static_cast<double>(n_ring) / static_cast<double>(cluster_pts.size());
}

double estimate_ground_from_workspace(
  const std::vector<pcl::PointXYZRGB> & pts,
  std::vector<pcl::PointXYZRGB> & roi_pts)
{
  roi_pts.clear();
  std::vector<float> z_vals;
  z_vals.reserve(pts.size());

  for (const auto & p : pts) {
    if (p.x < -0.75f || p.x > 0.75f || p.y < -0.55f || p.y > 0.55f) {
      continue;
    }
    if (p.z < -0.03f || p.z > 0.25f) {
      continue;
    }
    if (p.x > -0.10f && p.x < 0.10f && p.y > -0.12f && p.y < 0.12f) {
      continue;
    }
    roi_pts.push_back(p);
    z_vals.push_back(p.z);
  }

  if (z_vals.empty()) {
    return 0.0;
  }

  std::sort(z_vals.begin(), z_vals.end());
  const std::size_t idx = std::min<std::size_t>(
    static_cast<std::size_t>(z_vals.size() * 0.06), z_vals.size() - 1);
  return static_cast<double>(z_vals[idx]);
}

bool estimate_scan_reference_planes(
  const rclcpp::Logger & logger,
  const std::vector<pcl::PointXYZRGB> & roi_pts,
  double & table_z,
  double & board_z,
  bool & have_board_layer)
{
  table_z = 0.0;
  board_z = 0.0;
  have_board_layer = false;

  if (roi_pts.empty()) {
    return false;
  }

  std::vector<float> z_vals;
  z_vals.reserve(roi_pts.size());
  for (const auto & p : roi_pts) {
    z_vals.push_back(p.z);
  }
  std::sort(z_vals.begin(), z_vals.end());

  const std::size_t table_idx = std::min<std::size_t>(
    static_cast<std::size_t>(z_vals.size() * 0.06), z_vals.size() - 1);
  table_z = static_cast<double>(z_vals[table_idx]);

  const double z_min = static_cast<double>(z_vals.front());
  const double z_max = static_cast<double>(z_vals.back());
  if (z_max - z_min < 1e-4) {
    board_z = table_z;
    return true;
  }

  constexpr double k_bin_w = 0.004;
  const int bins = std::max(1, static_cast<int>(std::ceil((z_max - z_min) / k_bin_w)));
  std::vector<int> hist(static_cast<std::size_t>(bins), 0);

  for (float z : z_vals) {
    int bi = static_cast<int>(std::floor((static_cast<double>(z) - z_min) / k_bin_w));
    bi = std::clamp(bi, 0, bins - 1);
    hist[static_cast<std::size_t>(bi)] += 1;
  }

  int best_bin = -1;
  int best_count = 0;
  for (int bi = 0; bi < bins; ++bi) {
    const double zc = z_min + (static_cast<double>(bi) + 0.5) * k_bin_w;
    const double dz = zc - table_z;
    if (dz < 0.015 || dz > 0.070) {
      continue;
    }
    if (hist[static_cast<std::size_t>(bi)] > best_count) {
      best_count = hist[static_cast<std::size_t>(bi)];
      best_bin = bi;
    }
  }

  if (best_bin >= 0 && best_count > 200) {
    board_z = z_min + (static_cast<double>(best_bin) + 0.5) * k_bin_w;
    have_board_layer = true;
  } else {
    board_z = table_z;
    have_board_layer = false;
  }

  RCLCPP_WARN(
    logger,
    "[ScanGlobal] plane estimate: table_z=%.3f board_z=%.3f have_board=%s",
    table_z, board_z, have_board_layer ? "true" : "false");

  return true;
}

bool transform_cloud_to_world_points(
  const tf2_ros::Buffer & tf_buffer,
  const rclcpp::Logger & logger,
  const PointCPtr & cloud,
  const std::string & cloud_frame_id,
  std::vector<pcl::PointXYZRGB> & world_pts)
{
  world_pts.clear();
  if (!cloud || cloud->empty() || cloud_frame_id.empty()) {
    return false;
  }

  geometry_msgs::msg::TransformStamped tf_geom;
  try {
    tf_geom = tf_buffer.lookupTransform(
      "world", cloud_frame_id, rclcpp::Time(0), tf2::durationFromSec(2.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(logger, "[ScanGlobal] TF world<-cloud failed: %s", ex.what());
    return false;
  }

  tf2::Transform T_w_c;
  tf2::fromMsg(tf_geom.transform, T_w_c);

  world_pts.reserve(cloud->size() / 2 + 1);
  const std::size_t stride = std::max<std::size_t>(1, cloud->size() / 35000);

  for (std::size_t i = 0; i < cloud->size(); i += stride) {
    const auto & pt = cloud->points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }
    const tf2::Vector3 vc(pt.x, pt.y, pt.z);
    const tf2::Vector3 vw = T_w_c * vc;

    pcl::PointXYZRGB out;
    out.x = static_cast<float>(vw.x());
    out.y = static_cast<float>(vw.y());
    out.z = static_cast<float>(vw.z());
    out.r = pt.r;
    out.g = pt.g;
    out.b = pt.b;
    world_pts.push_back(out);
  }

  return !world_pts.empty();
}

std::vector<pcl::PointXYZRGB> downsample_world_points(
  const std::vector<pcl::PointXYZRGB> & in,
  double voxel)
{
  std::map<std::array<int, 3>, pcl::PointXYZRGB> voxels;
  for (const auto & p : in) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
      continue;
    }
    const std::array<int, 3> key = {
      static_cast<int>(std::floor(static_cast<double>(p.x) / voxel)),
      static_cast<int>(std::floor(static_cast<double>(p.y) / voxel)),
      static_cast<int>(std::floor(static_cast<double>(p.z) / voxel))
    };
    voxels.emplace(key, p);
  }

  std::vector<pcl::PointXYZRGB> out;
  out.reserve(voxels.size());
  for (const auto & kv : voxels) {
    out.push_back(kv.second);
  }
  return out;
}

std::vector<std::vector<int>> cluster_points_xy_z(
  const std::vector<pcl::PointXYZRGB> & pts,
  double xy_radius,
  double max_dz,
  std::size_t min_cluster_size)
{
  std::vector<std::vector<int>> clusters;
  if (pts.empty()) {
    return clusters;
  }

  const double xy_radius_sq = xy_radius * xy_radius;
  std::vector<bool> visited(pts.size(), false);

  for (std::size_t i = 0; i < pts.size(); ++i) {
    if (visited[i]) {
      continue;
    }

    visited[i] = true;
    std::vector<std::size_t> queue;
    queue.push_back(i);
    std::vector<int> cluster;
    cluster.push_back(static_cast<int>(i));

    for (std::size_t q = 0; q < queue.size(); ++q) {
      const auto & a = pts[queue[q]];
      for (std::size_t j = 0; j < pts.size(); ++j) {
        if (visited[j]) {
          continue;
        }

        const auto & b = pts[j];
        const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
        const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
        const double dz = std::fabs(static_cast<double>(a.z) - static_cast<double>(b.z));

        if (dx * dx + dy * dy <= xy_radius_sq && dz <= max_dz) {
          visited[j] = true;
          queue.push_back(j);
          cluster.push_back(static_cast<int>(j));
        }
      }
    }

    if (cluster.size() >= min_cluster_size) {
      clusters.push_back(std::move(cluster));
    }
  }

  return clusters;
}

Task3MergedScanObject build_object_from_cluster(
  const std::vector<pcl::PointXYZRGB> & cluster_pts,
  const std::string & category)
{
  Task3MergedScanObject obj;
  obj.category = category;
  obj.vis_points = cluster_pts;
  obj.seen_count = 1;
  obj.cross_votes = 0;
  obj.nought_votes = 0;
  obj.unknown_votes = 0;
  obj.best_view_category = category;

  double min_x = std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double min_z = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  double max_z = -std::numeric_limits<double>::infinity();

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  double sum_r = 0.0;
  double sum_g = 0.0;
  double sum_b = 0.0;

  for (const auto & p : cluster_pts) {
    const double x = static_cast<double>(p.x);
    const double y = static_cast<double>(p.y);
    const double z = static_cast<double>(p.z);

    min_x = std::min(min_x, x);
    min_y = std::min(min_y, y);
    min_z = std::min(min_z, z);
    max_x = std::max(max_x, x);
    max_y = std::max(max_y, y);
    max_z = std::max(max_z, z);

    sum_x += x;
    sum_y += y;
    sum_z += z;
    sum_r += p.r;
    sum_g += p.g;
    sum_b += p.b;
  }

  const double n = static_cast<double>(cluster_pts.size());
  obj.position.x = sum_x / n;
  obj.position.y = sum_y / n;
  obj.position.z = sum_z / n;
  obj.top_z = max_z;

  obj.size_x = max_x - min_x;
  obj.size_y = max_y - min_y;
  obj.size_z = max_z - min_z;

  obj.mean_r = sum_r / n;
  obj.mean_g = sum_g / n;
  obj.mean_b = sum_b / n;

  obj.point_count = cluster_pts.size();
  obj.best_view_position = obj.position;
  obj.best_view_size_x = obj.size_x;
  obj.best_view_size_y = obj.size_y;
  obj.best_view_size_z = obj.size_z;

  return obj;
}

void compute_pca_xy_metrics(
  const std::vector<pcl::PointXYZRGB> & cluster_pts,
  const geometry_msgs::msg::Point & center,
  double & pca_ratio,
  double & axis_balance,
  double & arm_fill_score)
{
  pca_ratio = std::numeric_limits<double>::infinity();
  axis_balance = 0.0;
  arm_fill_score = 0.0;
  if (cluster_pts.size() < 8) {
    return;
  }

  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (const auto & p : cluster_pts) {
    const double dx = static_cast<double>(p.x) - center.x;
    const double dy = static_cast<double>(p.y) - center.y;
    cov(0, 0) += dx * dx;
    cov(0, 1) += dx * dy;
    cov(1, 0) += dx * dy;
    cov(1, 1) += dy * dy;
  }
  cov /= std::max<double>(1.0, static_cast<double>(cluster_pts.size()));

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(cov);
  if (solver.info() != Eigen::Success) {
    return;
  }

  const Eigen::Vector2d evals = solver.eigenvalues();
  const Eigen::Matrix2d evecs = solver.eigenvectors();
  const double l0 = std::max(1e-9, static_cast<double>(evals(1)));
  const double l1 = std::max(1e-9, static_cast<double>(evals(0)));
  pca_ratio = std::sqrt(l0 / l1);

  const Eigen::Vector2d major = evecs.col(1);
  const Eigen::Vector2d minor = evecs.col(0);

  double max_u = 0.0;
  double max_v = 0.0;
  int major_band = 0;
  int minor_band = 0;
  for (const auto & p : cluster_pts) {
    const double dx = static_cast<double>(p.x) - center.x;
    const double dy = static_cast<double>(p.y) - center.y;
    const Eigen::Vector2d d(dx, dy);
    const double u = std::fabs(d.dot(major));
    const double v = std::fabs(d.dot(minor));
    max_u = std::max(max_u, u);
    max_v = std::max(max_v, v);
  }

  const double band_u = std::max(0.012, 0.22 * max_u);
  const double band_v = std::max(0.012, 0.22 * max_v);
  for (const auto & p : cluster_pts) {
    const double dx = static_cast<double>(p.x) - center.x;
    const double dy = static_cast<double>(p.y) - center.y;
    const Eigen::Vector2d d(dx, dy);
    const double u = std::fabs(d.dot(major));
    const double v = std::fabs(d.dot(minor));
    if (v <= band_v) {
      ++major_band;
    }
    if (u <= band_u) {
      ++minor_band;
    }
  }

  const double denom = std::max(1.0, static_cast<double>(cluster_pts.size()));
  const double major_fill = static_cast<double>(major_band) / denom;
  const double minor_fill = static_cast<double>(minor_band) / denom;
  axis_balance = std::min(major_fill, minor_fill) / std::max(1e-6, std::max(major_fill, minor_fill));
  arm_fill_score = std::min(major_fill, minor_fill);
}

void compute_cross_arm_metrics(
  const std::vector<pcl::PointXYZRGB> & pts,
  const geometry_msgs::msg::Point & center,
  double & arm_fill,
  double & axis_balance,
  double & pca_ratio)
{
  arm_fill = 0.0;
  axis_balance = 0.0;
  pca_ratio = 1.0;

  if (pts.size() < 20) {
    return;
  }

  double sxx = 0.0;
  double sxy = 0.0;
  double syy = 0.0;

  for (const auto & p : pts) {
    const double x = static_cast<double>(p.x) - center.x;
    const double y = static_cast<double>(p.y) - center.y;
    sxx += x * x;
    sxy += x * y;
    syy += y * y;
  }

  const double tr = sxx + syy;
  const double disc = std::sqrt(std::max(0.0, (sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy));
  const double l1 = 0.5 * (tr + disc);
  const double l2 = 0.5 * (tr - disc);

  pca_ratio = std::sqrt(std::max(l1, 1e-9) / std::max(l2, 1e-9));

  double ux = 1.0;
  double uy = 0.0;
  if (std::fabs(sxy) > 1e-9 || std::fabs(l1 - sxx) > 1e-9) {
    ux = sxy;
    uy = l1 - sxx;
    const double n = std::hypot(ux, uy);
    if (n > 1e-9) {
      ux /= n;
      uy /= n;
    }
  }

  const double vx = -uy;
  const double vy = ux;

  std::vector<std::pair<double, double>> uv;
  uv.reserve(pts.size());

  double max_u = 0.0;
  double max_v = 0.0;

  for (const auto & p : pts) {
    const double x = static_cast<double>(p.x) - center.x;
    const double y = static_cast<double>(p.y) - center.y;
    const double u = x * ux + y * uy;
    const double v = x * vx + y * vy;
    uv.emplace_back(u, v);
    max_u = std::max(max_u, std::fabs(u));
    max_v = std::max(max_v, std::fabs(v));
  }

  if (max_u < 1e-6 || max_v < 1e-6) {
    return;
  }

  int arm_u_count = 0;
  int arm_v_count = 0;
  for (const auto & q : uv) {
    const double u = q.first;
    const double v = q.second;
    if (std::fabs(v) < 0.18 * max_v && std::fabs(u) < 0.90 * max_u) {
      ++arm_u_count;
    }
    if (std::fabs(u) < 0.18 * max_u && std::fabs(v) < 0.90 * max_v) {
      ++arm_v_count;
    }
  }

  const double fu = static_cast<double>(arm_u_count) / static_cast<double>(pts.size());
  const double fv = static_cast<double>(arm_v_count) / static_cast<double>(pts.size());
  arm_fill = std::min(fu, fv);
  axis_balance = std::min(fu, fv) / std::max(1e-6, std::max(fu, fv));
}

std::string classify_shape_cluster(
  const rclcpp::Logger & logger,
  const std::vector<pcl::PointXYZRGB> & cluster_pts,
  const geometry_msgs::msg::Point & center,
  double size_x,
  double size_y,
  double size_z,
  double mean_r,
  double mean_g,
  double mean_b)
{
  if (cluster_pts.size() < 30) {
    return "shape_unknown";
  }

  double center_fill_ratio = 1.0;
  double ring_fill_ratio = 0.0;
  double max_r_metric = 0.0;
  compute_hollow_metrics(cluster_pts, center, center_fill_ratio, ring_fill_ratio, max_r_metric);

  double arm_fill = 0.0;
  double axis_balance = 0.0;
  double pca_ratio = 1.0;
  compute_cross_arm_metrics(cluster_pts, center, arm_fill, axis_balance, pca_ratio);

  const double max_rgb = std::max({mean_r, mean_g, mean_b});
  const double min_rgb = std::min({mean_r, mean_g, mean_b});
  const double long_side = std::max(size_x, size_y);
  const double short_side = std::max(1e-6, std::min(size_x, size_y));
  const double aspect = long_side / short_side;

  const bool basket_warm =
    mean_r > 45.0 && mean_g > 16.0 && mean_b > 16.0 &&
    mean_r > mean_g + 16.0 && mean_r > mean_b + 16.0;
  const bool basket_big = size_x > 0.30 && size_y > 0.30;
  const bool basket_low = size_z > 0.015 && size_z < 0.10;
  const bool basket_hollow = center_fill_ratio < 0.08 && ring_fill_ratio > 0.45;
  if (basket_warm && basket_big && basket_low && basket_hollow) {
    RCLCPP_WARN(
      logger,
      "[ScanGlobal] shape metrics: center_fill=%.2f ring_fill=%.2f pca_ratio=%.2f axis_balance=%.2f arm_fill=%.2f aspect=%.2f -> basket",
      center_fill_ratio, ring_fill_ratio, pca_ratio, axis_balance, arm_fill, aspect);
    return "basket";
  }

  const bool obstacle_black = max_rgb < 30.0;
  const bool obstacle_neutral = (max_rgb - min_rgb) < 12.0;
  const bool obstacle_size_ok =
    size_x > 0.015 && size_x < 0.14 &&
    size_y > 0.015 && size_y < 0.14 &&
    size_z < 0.16;
  if (obstacle_black && obstacle_neutral && obstacle_size_ok) {
    RCLCPP_WARN(
      logger,
      "[ScanGlobal] shape metrics: center_fill=%.2f ring_fill=%.2f pca_ratio=%.2f axis_balance=%.2f arm_fill=%.2f aspect=%.2f -> obstacle",
      center_fill_ratio, ring_fill_ratio, pca_ratio, axis_balance, arm_fill, aspect);
    return "obstacle";
  }

  const bool shape_size_gate =
    long_side >= 0.08 && long_side <= 0.31 &&
    short_side >= 0.06 && short_side <= 0.31 &&
    size_z > 0.015 && size_z < 0.06;
  if (!shape_size_gate) {
    RCLCPP_WARN(
      logger,
      "[ScanGlobal] reject cluster: points=%zu pos=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f) rgb=(%.1f, %.1f, %.1f) reason=shape_size_gate warm=%s dark=%s center_fill=%.2f ring_fill=%.2f pca_ratio=%.2f axis_balance=%.2f arm_fill=%.2f",
      cluster_pts.size(), center.x, center.y, center.z, size_x, size_y, size_z,
      mean_r, mean_g, mean_b, basket_warm ? "true" : "false",
      (obstacle_black && obstacle_neutral) ? "true" : "false",
      center_fill_ratio, ring_fill_ratio, pca_ratio, axis_balance, arm_fill);
    return "shape_unknown";
  }

  const bool nought_square = aspect < 1.35;
  const bool nought_hollow =
    center_fill_ratio < 0.06 && ring_fill_ratio > 0.50 &&
    ring_fill_ratio > center_fill_ratio * 3.5;
  const bool nought_roundish = pca_ratio < 1.30;
  const bool is_strict_nought = nought_square && nought_hollow && nought_roundish;

  const bool cross_like =
    arm_fill > 0.16 && axis_balance > 0.45 && pca_ratio > 1.00 && center_fill_ratio > 0.06;
  if (cross_like && !is_strict_nought) {
    RCLCPP_WARN(
      logger,
      "[ScanGlobal] shape metrics: center_fill=%.2f ring_fill=%.2f pca_ratio=%.2f axis_balance=%.2f arm_fill=%.2f aspect=%.2f -> cross",
      center_fill_ratio, ring_fill_ratio, pca_ratio, axis_balance, arm_fill, aspect);
    return "cross";
  }

  if (is_strict_nought) {
    RCLCPP_WARN(
      logger,
      "[ScanGlobal] shape metrics: center_fill=%.2f ring_fill=%.2f pca_ratio=%.2f axis_balance=%.2f arm_fill=%.2f aspect=%.2f -> nought",
      center_fill_ratio, ring_fill_ratio, pca_ratio, axis_balance, arm_fill, aspect);
    return "nought";
  }

  RCLCPP_WARN(
    logger,
    "[ScanGlobal] shape metrics: center_fill=%.2f ring_fill=%.2f pca_ratio=%.2f axis_balance=%.2f arm_fill=%.2f aspect=%.2f -> cross_fallback",
    center_fill_ratio, ring_fill_ratio, pca_ratio, axis_balance, arm_fill, aspect);
  return "cross";
}

void suppress_basket_neighbors_on_final(std::vector<Task3MergedScanObject> & objects)
{
  std::vector<bool> remove(objects.size(), false);

  for (std::size_t i = 0; i < objects.size(); ++i) {
    if (objects[i].category != "basket") {
      continue;
    }

    const auto & basket = objects[i];
    const double basket_hx = 0.5 * std::max(basket.size_x, 0.36) + 0.05;
    const double basket_hy = 0.5 * std::max(basket.size_y, 0.36) + 0.05;

    for (std::size_t j = 0; j < objects.size(); ++j) {
      if (i == j || remove[j] || objects[j].category == "basket") {
        continue;
      }
      const auto & other = objects[j];
      const bool center_inside =
        std::fabs(other.position.x - basket.position.x) <= basket_hx &&
        std::fabs(other.position.y - basket.position.y) <= basket_hy;
      if (center_inside) {
        remove[j] = true;
      }
    }
  }

  std::vector<Task3MergedScanObject> filtered;
  filtered.reserve(objects.size());
  for (std::size_t i = 0; i < objects.size(); ++i) {
    if (!remove[i]) {
      filtered.push_back(std::move(objects[i]));
    }
  }
  objects.swap(filtered);
}

std::vector<Task3MergedScanObject> detect_task3_objects_from_world_points(
  const rclcpp::Logger & logger,
  const std::vector<pcl::PointXYZRGB> & world_pts)
{
  std::vector<Task3MergedScanObject> objects;
  if (world_pts.empty()) {
    return objects;
  }

  std::vector<pcl::PointXYZRGB> roi_pts;
  const double global_ground_z = estimate_ground_from_workspace(world_pts, roi_pts);

  double table_z = global_ground_z;
  double board_z = global_ground_z;
  bool have_board_layer = false;
  estimate_scan_reference_planes(logger, roi_pts, table_z, board_z, have_board_layer);

  constexpr double k_above_plane_low = 0.010;
  constexpr double k_above_plane_high = 0.140;
  std::vector<pcl::PointXYZRGB> elevated_pts;
  elevated_pts.reserve(roi_pts.size());

  for (const auto & p : roi_pts) {
    const double r = std::hypot(static_cast<double>(p.x), static_cast<double>(p.y));
    if (r < 0.16 || r > 0.72) {
      continue;
    }

    const double dz_table = static_cast<double>(p.z) - table_z;
    const double dz_board = static_cast<double>(p.z) - board_z;
    bool elevated = false;
    if (have_board_layer) {
      const double ref_dz = (std::fabs(dz_board) < std::fabs(dz_table)) ? dz_board : dz_table;
      elevated = (ref_dz >= k_above_plane_low && ref_dz <= k_above_plane_high);
    } else {
      elevated = (dz_table >= k_above_plane_low && dz_table <= k_above_plane_high);
    }
    if (elevated) {
      elevated_pts.push_back(p);
    }
  }

  RCLCPP_WARN(
    logger,
    "[ScanGlobal] roi_pts=%zu elevated=%zu global_ground=%.3f table_z=%.3f board_z=%.3f have_board=%s",
    roi_pts.size(), elevated_pts.size(), global_ground_z, table_z, board_z,
    have_board_layer ? "true" : "false");

  const auto all_clusters = cluster_points_xy_z(elevated_pts, 0.022, 0.024, 12);
  RCLCPP_WARN(logger, "[ScanGlobal] unified_clusters=%zu", all_clusters.size());

  for (const auto & cluster : all_clusters) {
    std::vector<pcl::PointXYZRGB> cluster_pts;
    cluster_pts.reserve(cluster.size());
    for (int idx : cluster) {
      cluster_pts.push_back(elevated_pts[static_cast<std::size_t>(idx)]);
    }

    auto obj = build_object_from_cluster(cluster_pts, "unknown");
    const double max_rgb = std::max({obj.mean_r, obj.mean_g, obj.mean_b});
    const double min_rgb = std::min({obj.mean_r, obj.mean_g, obj.mean_b});
    const double long_side = std::max(obj.size_x, obj.size_y);
    const double short_side = std::min(obj.size_x, obj.size_y);
    const double aspect = long_side / std::max(1e-6, short_side);

    double center_fill_ratio = 1.0;
    double ring_fill_ratio = 0.0;
    double max_r_metric = 0.0;
    compute_hollow_metrics(cluster_pts, obj.position, center_fill_ratio, ring_fill_ratio, max_r_metric);

    const bool warm_like =
      is_warm_red_point(cluster_pts.front()) ||
      (obj.mean_r > 45.0 && obj.mean_r > obj.mean_g + 18.0 && obj.mean_r > obj.mean_b + 18.0);
    const bool dark_like = max_rgb < 36.0 && (max_rgb - min_rgb) < 14.0;

    const bool basket_warm =
      obj.mean_r > 45.0 && obj.mean_g > 16.0 && obj.mean_b > 16.0 &&
      obj.mean_r > obj.mean_g + 26.0 && obj.mean_r > obj.mean_b + 26.0 &&
      obj.mean_g < 40.0 && obj.mean_b < 40.0;
    const bool basket_big = long_side > 0.32 && short_side > 0.30;
    const bool basket_partial_big = long_side > 0.34 && short_side > 0.28;
    const bool basket_low = obj.size_z > 0.010 && obj.size_z < 0.12;
    const bool basket_hollow = center_fill_ratio < 0.10 && ring_fill_ratio > 0.12;
    const bool basket_rim = looks_like_rim_only_basket(
      obj.size_x, obj.size_y, obj.size_z, obj.mean_r, obj.mean_g, obj.mean_b,
      center_fill_ratio, ring_fill_ratio);

    if (basket_warm && basket_low && ((basket_big && basket_hollow) ||
      (basket_partial_big && basket_hollow) || basket_rim))
    {
      append_local_roi_points(
        roi_pts, obj.position, std::max(obj.size_x, 0.36), std::max(obj.size_y, 0.36),
        table_z + k_basket_z_min_offset, table_z + k_basket_z_max_offset,
        k_basket_pad_x, k_basket_pad_y, true, false, obj.vis_points);
      obj = build_object_from_cluster(obj.vis_points, "basket");
      objects.push_back(std::move(obj));
      continue;
    }

    const bool obstacle_black = max_rgb < 32.0;
    const bool obstacle_neutral = (max_rgb - min_rgb) < 14.0;
    const bool obstacle_size_ok =
      obj.size_x > 0.015 && obj.size_x < 0.16 &&
      obj.size_y > 0.015 && obj.size_y < 0.16 &&
      obj.size_z < 0.14;
    if (dark_like && obstacle_black && obstacle_neutral && obstacle_size_ok) {
      append_local_roi_points(
        roi_pts, obj.position, std::max(obj.size_x, 0.08), std::max(obj.size_y, 0.08),
        table_z + 0.004, table_z + 0.100, 0.020, 0.020, false, true, obj.vis_points);
      obj = build_object_from_cluster(obj.vis_points, "obstacle");
      objects.push_back(std::move(obj));
      continue;
    }

    if (long_side > 0.31 || short_side < 0.028 || obj.size_z > 0.10) {
      RCLCPP_WARN(
        logger,
        "[ScanGlobal] reject cluster: points=%zu pos=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f) rgb=(%.1f, %.1f, %.1f) reason=shape_size_gate warm=%s dark=%s center_fill=%.2f ring_fill=%.2f",
        cluster_pts.size(), obj.position.x, obj.position.y, obj.position.z,
        obj.size_x, obj.size_y, obj.size_z, obj.mean_r, obj.mean_g, obj.mean_b,
        warm_like ? "true" : "false", dark_like ? "true" : "false",
        center_fill_ratio, ring_fill_ratio);
      continue;
    }

    if (aspect >= 2.6 && cluster_pts.size() < 120) {
      RCLCPP_WARN(
        logger,
        "[ScanGlobal] reject cluster: points=%zu pos=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f) reason=thin_long_fragment aspect=%.2f",
        cluster_pts.size(), obj.position.x, obj.position.y, obj.position.z,
        obj.size_x, obj.size_y, obj.size_z, aspect);
      continue;
    }

    obj.category = classify_shape_cluster(
      logger, cluster_pts, obj.position, obj.size_x, obj.size_y, obj.size_z,
      obj.mean_r, obj.mean_g, obj.mean_b);

    if (obj.category == "cross") {
      obj.cross_votes = 1;
      objects.push_back(std::move(obj));
    } else if (obj.category == "nought") {
      obj.nought_votes = 1;
      objects.push_back(std::move(obj));
    } else if (obj.category == "basket" || obj.category == "obstacle") {
      objects.push_back(std::move(obj));
    } else {
      double pca_ratio = std::numeric_limits<double>::infinity();
      double axis_balance = 0.0;
      double arm_fill_score = 0.0;
      compute_pca_xy_metrics(cluster_pts, obj.position, pca_ratio, axis_balance, arm_fill_score);
      RCLCPP_WARN(
        logger,
        "[ScanGlobal] reject cluster: points=%zu pos=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f) rgb=(%.1f, %.1f, %.1f) reason=shape_unknown center_fill=%.2f ring_fill=%.2f pca_ratio=%.2f axis_balance=%.2f arm_fill=%.2f warm=%s dark=%s",
        cluster_pts.size(), obj.position.x, obj.position.y, obj.position.z,
        obj.size_x, obj.size_y, obj.size_z, obj.mean_r, obj.mean_g, obj.mean_b,
        center_fill_ratio, ring_fill_ratio, pca_ratio, axis_balance, arm_fill_score,
        warm_like ? "true" : "false", dark_like ? "true" : "false");
    }
  }

  suppress_basket_neighbors_on_final(objects);
  return objects;
}

void write_task3_debug_outputs(
  const rclcpp::Logger & logger,
  const std::vector<Task3MergedScanObject> & merged)
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  std::size_t reserve_points = 0;
  for (const auto & obj : merged) {
    reserve_points += std::max<std::size_t>(obj.vis_points.size(), 1);
  }
  cloud.reserve(reserve_points);

  for (const auto & obj : merged) {
    std::uint8_t rr = 230, gg = 200, bb = 40;
    if (obj.category == "cross") { rr = 40; gg = 80; bb = 255; }
    else if (obj.category == "nought") { rr = 40; gg = 200; bb = 80; }
    else if (obj.category == "obstacle") { rr = 255; gg = 60; bb = 255; }
    else if (obj.category == "basket") { rr = 170; gg = 80; bb = 80; }

    if (obj.vis_points.empty()) {
      pcl::PointXYZRGB p;
      p.x = static_cast<float>(obj.position.x);
      p.y = static_cast<float>(obj.position.y);
      p.z = static_cast<float>(obj.position.z);
      p.r = rr; p.g = gg; p.b = bb;
      cloud.push_back(p);
      continue;
    }

    for (const auto & src_pt : obj.vis_points) {
      pcl::PointXYZRGB p = src_pt;
      p.r = rr; p.g = gg; p.b = bb;
      cloud.push_back(p);
    }
  }

  cloud.width = static_cast<std::uint32_t>(cloud.size());
  cloud.height = 1;
  cloud.is_dense = true;
  const std::string pcd_path = "/tmp/task3_result.pcd";
  pcl::io::savePCDFileBinary(pcd_path, cloud);

  const std::string csv_path = "/tmp/task3_scan_summary.csv";
  std::ofstream csv(csv_path);
  csv << "category,x,y,z,size_x,size_y,size_z,r,g,b,seen,cross_votes,nought_votes,unknown_votes\n";
  for (const auto & obj : merged) {
    csv << obj.category << ","
        << obj.position.x << "," << obj.position.y << "," << obj.position.z << ","
        << obj.size_x << "," << obj.size_y << "," << obj.size_z << ","
        << obj.mean_r << "," << obj.mean_g << "," << obj.mean_b << ","
        << obj.seen_count << "," << obj.cross_votes << ","
        << obj.nought_votes << "," << obj.unknown_votes << "\n";
  }

  RCLCPP_WARN(logger, "[Task3Scan] wrote debug outputs: %s , %s", pcd_path.c_str(), csv_path.c_str());
}

void write_task3_fused_cloud_debug_outputs(
  const rclcpp::Logger & logger,
  const std::vector<pcl::PointXYZRGB> & fused_world_pts,
  const std::vector<Task3MergedScanObject> & objects)
{
  pcl::PointCloud<pcl::PointXYZRGB> full_cloud;
  full_cloud.reserve(fused_world_pts.size());
  for (const auto & p : fused_world_pts) {
    full_cloud.push_back(p);
  }
  full_cloud.width = static_cast<std::uint32_t>(full_cloud.size());
  full_cloud.height = 1;
  full_cloud.is_dense = true;

  const std::string full_pcd_path = "/tmp/task3_fused_world.pcd";
  pcl::io::savePCDFileBinary(full_pcd_path, full_cloud);

  const std::string csv_path = "/tmp/task3_fused_world_points.csv";
  std::ofstream csv(csv_path);
  csv << "x,y,z,r,g,b\n";
  for (const auto & p : fused_world_pts) {
    csv << p.x << "," << p.y << "," << p.z << ","
        << static_cast<int>(p.r) << ","
        << static_cast<int>(p.g) << ","
        << static_cast<int>(p.b) << "\n";
  }

  RCLCPP_WARN(logger, "[Task3Scan] wrote fused debug outputs: %s , %s (objects=%zu)",
    full_pcd_path.c_str(), csv_path.c_str(), objects.size());
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
  loc_world.header.frame_id = object_loc.header.frame_id.empty() ? "world" : object_loc.header.frame_id;
  loc_world.header.stamp = rclcpp::Time(0);
  loc_world.point = object_loc.point;

  try {
    const auto lw = tf_buffer.transform(loc_world, "world", tf2::durationFromSec(2.0));
    ox = lw.point.x;
    oy = lw.point.y;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_DEBUG(logger, "[Task1] object point TF fallback (assume world coords): %s", ex.what());
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
  const std::size_t gi = std::min(
    static_cast<std::size_t>(ground_z_samples.size() * 0.06), ground_z_samples.size() - 1);
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
    const double score = min_radius[i] + min_radius[i90] + min_radius[i180] + min_radius[i270];
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

bool estimate_task1_localized_pick_from_cloud(
  const tf2_ros::Buffer & tf_buffer,
  const rclcpp::Logger & logger,
  const PointCPtr & cloud,
  const std::string & cloud_frame_id,
  const geometry_msgs::msg::PointStamped & nominal_object_loc,
  const std::string & shape_type,
  geometry_msgs::msg::Point & detected_center_world,
  double & detected_top_z_world,
  double & estimated_yaw_rad)
{
  detected_center_world = nominal_object_loc.point;
  detected_top_z_world = nominal_object_loc.point.z;
  estimated_yaw_rad = 0.0;

  std::vector<std::pair<double, double>> xy_pts;
  double ox = 0.0;
  double oy = 0.0;
  double ground_z = 0.0;

  if (!collect_task1_object_top_points(
        tf_buffer, logger, cloud, cloud_frame_id, nominal_object_loc,
        xy_pts, ox, oy, ground_z))
  {
    return false;
  }

  geometry_msgs::msg::TransformStamped tf_geom;
  try {
    tf_geom = tf_buffer.lookupTransform(
      "world", cloud_frame_id, rclcpp::Time(0), tf2::durationFromSec(2.0));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(logger, "[Task1] localize TF world<-cloud failed: %s", ex.what());
    return false;
  }

  tf2::Transform T_w_c;
  tf2::fromMsg(tf_geom.transform, T_w_c);

  geometry_msgs::msg::PointStamped loc_world;
  loc_world.header.frame_id =
    nominal_object_loc.header.frame_id.empty() ? "world" : nominal_object_loc.header.frame_id;
  loc_world.header.stamp = rclcpp::Time(0);
  loc_world.point = nominal_object_loc.point;

  try {
    const auto lw = tf_buffer.transform(loc_world, "world", tf2::durationFromSec(2.0));
    ox = lw.point.x;
    oy = lw.point.y;
  } catch (const tf2::TransformException &) {
  }

  const std::size_t n = cloud->size();
  const std::size_t stride = std::max<std::size_t>(1, n / 50000);

  const double z_min = ground_z + 0.012;
  constexpr double k_object_crop_radius = 0.11;

  std::vector<geometry_msgs::msg::Point> obj_pts_world;
  obj_pts_world.reserve(n / stride + 1);

  for (std::size_t i = 0; i < n; i += stride) {
    const auto & pt = cloud->points[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      continue;
    }

    const tf2::Vector3 vc(pt.x, pt.y, pt.z);
    const tf2::Vector3 vw = T_w_c * vc;

    if (vw.z() < z_min) {
      continue;
    }

    const double dx = vw.x() - ox;
    const double dy = vw.y() - oy;
    if (std::hypot(dx, dy) <= k_object_crop_radius) {
      geometry_msgs::msg::Point p;
      p.x = vw.x();
      p.y = vw.y();
      p.z = vw.z();
      obj_pts_world.push_back(p);
    }
  }

  if (obj_pts_world.size() < 40) {
    RCLCPP_WARN(logger, "[Task1] localize failed: not enough 3d object points");
    return false;
  }

  double sum_x = 0.0;
  double sum_y = 0.0;
  std::vector<double> z_vals;
  z_vals.reserve(obj_pts_world.size());

  for (const auto & p : obj_pts_world) {
    sum_x += p.x;
    sum_y += p.y;
    z_vals.push_back(p.z);
  }

  std::sort(z_vals.begin(), z_vals.end());
  const std::size_t zi = std::min(
    static_cast<std::size_t>(0.90 * static_cast<double>(z_vals.size() - 1)),
    z_vals.size() - 1);
  detected_top_z_world = z_vals[zi];

  double top_sum_x = 0.0;
  double top_sum_y = 0.0;
  int top_n = 0;
  for (const auto & p : obj_pts_world) {
    if (p.z >= detected_top_z_world - 0.010) {
      top_sum_x += p.x;
      top_sum_y += p.y;
      ++top_n;
    }
  }

  if (top_n > 0) {
    detected_center_world.x = top_sum_x / static_cast<double>(top_n);
    detected_center_world.y = top_sum_y / static_cast<double>(top_n);
  } else {
    detected_center_world.x = sum_x / static_cast<double>(obj_pts_world.size());
    detected_center_world.y = sum_y / static_cast<double>(obj_pts_world.size());
  }
  detected_center_world.z = detected_top_z_world;

  bool have_yaw = false;
  if (shape_type == "nought") {
    have_yaw = estimate_nought_inner_edge_yaw_impl(logger, xy_pts, estimated_yaw_rad);
  } else {
    have_yaw = estimate_cross_pca_yaw_impl(logger, xy_pts, estimated_yaw_rad);
  }

  if (!have_yaw) {
    estimated_yaw_rad = 0.0;
  }

  RCLCPP_INFO(
    logger,
    "[Task1] relocalized pick: center=(%.3f, %.3f, %.3f), yaw=%.3f rad (%.1f deg), top_z=%.3f, samples=%zu",
    detected_center_world.x,
    detected_center_world.y,
    detected_center_world.z,
    estimated_yaw_rad,
    estimated_yaw_rad * 180.0 / M_PI,
    detected_top_z_world,
    obj_pts_world.size());

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
  pointcloud_qos_reliable_ = node_->declare_parameter<bool>("pointcloud_qos_reliable", true);

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

  pointcloud_callback_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
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

geometry_msgs::msg::Quaternion cw2::make_top_down_q() const
{
  tf2::Quaternion q;
  q.setRPY(3.14159265358979323846, 0.0, -0.78539816339744830962);
  q.normalize();
  geometry_msgs::msg::Quaternion msg;
  tf2::convert(q, msg);
  return msg;
}

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
  const std::string src = object_loc.header.frame_id.empty() ? "world" : object_loc.header.frame_id;
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
  loc_world.header.frame_id = object_loc.header.frame_id.empty() ? "world" : object_loc.header.frame_id;
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
  const std::size_t gi = std::min(
    static_cast<std::size_t>(ground_z_samples.size() * 0.06), ground_z_samples.size() - 1);
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
  } else if (ring >= 120 && inner <= 20 && ratio >= task2_ring_inner_ratio_nought_min_) {
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

  RCLCPP_WARN(
    node_->get_logger(),
    "[CARTESIAN %s] planning_frame=%s eef_link=%s "
    "target_pos=(%.4f, %.4f, %.4f) target_q=(%.4f, %.4f, %.4f, %.4f) "
    "eef_step=%.4f min_fraction=%.2f",
    label,
    arm_group_->getPlanningFrame().c_str(),
    arm_group_->getEndEffectorLink().c_str(),
    target_pose.position.x,
    target_pose.position.y,
    target_pose.position.z,
    target_pose.orientation.x,
    target_pose.orientation.y,
    target_pose.orientation.z,
    target_pose.orientation.w,
    safe_eef_step,
    cartesian_min_fraction_);

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory traj;
  const double fraction =
    arm_group_->computeCartesianPath(waypoints, safe_eef_step, jump_threshold, traj);

  RCLCPP_WARN(
    node_->get_logger(),
    "[CARTESIAN %s] computeCartesianPath -> fraction=%.4f traj_points=%zu joint_names=%zu",
    label,
    fraction,
    traj.joint_trajectory.points.size(),
    traj.joint_trajectory.joint_names.size());

  if (!traj.joint_trajectory.points.empty()) {
    const auto & last_pt = traj.joint_trajectory.points.back();
    std::ostringstream oss;
    oss << "[CARTESIAN " << label << "] last_point positions=[";
    for (std::size_t i = 0; i < last_pt.positions.size(); ++i) {
      oss << last_pt.positions[i];
      if (i + 1 < last_pt.positions.size()) {
        oss << ", ";
      }
    }
    oss << "]";
    RCLCPP_WARN(node_->get_logger(), "%s", oss.str().c_str());
  }

  if (fraction + 1e-6 >= cartesian_min_fraction_) {
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = traj;

    const bool execution_ok =
      (arm_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS);

    if (!execution_ok) {
      RCLCPP_WARN(
        node_->get_logger(),
        "[CARTESIAN %s] execute failed; fallback to joint motion",
        label);
      return move_arm_to_pose_joint(target_pose);
    }

    RCLCPP_WARN(
      node_->get_logger(),
      "[CARTESIAN %s] execute success",
      label);
    return true;
  }

  RCLCPP_WARN(
    node_->get_logger(),
    "[CARTESIAN %s] incomplete (fraction=%.4f < %.2f), fallback to joint motion",
    label,
    fraction,
    cartesian_min_fraction_);

  return move_arm_to_pose_joint(target_pose);
}

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

  geometry_msgs::msg::Point localized_obj_world = object_point.point;
  double localized_top_z_world = object_point.point.z;
  double object_world_yaw = 0.0;
  bool have_localized_pose = false;
  bool have_yaw = false;

  if (reached_observation) {
    rclcpp::sleep_for(std::chrono::milliseconds(350));

    PointCPtr cloud;
    std::string cloud_frame_id;
    if (capture_latest_cloud(cloud, cloud_frame_id, 2.0)) {
      have_localized_pose = estimate_task1_localized_pick_from_cloud(
        tf_buffer_,
        node_->get_logger(),
        cloud,
        cloud_frame_id,
        object_point,
        shape_type,
        localized_obj_world,
        localized_top_z_world,
        object_world_yaw);

      have_yaw = have_localized_pose;
    } else {
      RCLCPP_WARN(node_->get_logger(), "[Task1] could not capture cloud for relocalization");
    }
  } else {
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task1] observation move failed; skip relocalization and use nominal object point");
  }

  if (!have_localized_pose) {
    localized_obj_world = object_point.point;
    localized_top_z_world = object_point.point.z;
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task1] relocalization failed; using nominal object point");
  }

  if (!have_yaw) {
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task1] yaw estimate failed; falling back to fixed orientation for shape '%s'",
      shape_type.c_str());
  }

  const geometry_msgs::msg::Quaternion grasp_q =
    have_yaw ? ee_orientation_for_shape_with_world_yaw(shape_type, object_world_yaw)
             : ee_orientation_for_shape(shape_type);

  // =========================
  // Shape-aware XY offset
  // =========================
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

  // 最终抓取点：重定位中心 + offset
  geometry_msgs::msg::Point pick_world = localized_obj_world;
  pick_world.x += off_x;
  pick_world.y += off_y;
  pick_world.z = localized_obj_world.z;

  // 放置点也应用同样 offset，保证相对抓取方式一致
  geometry_msgs::msg::Point goal_world = goal_point.point;
  goal_world.x += off_x;
  goal_world.y += off_y;

  const geometry_msgs::msg::Point pick_p =
    transform_point_to_planning_frame(pick_world, "world");
  const geometry_msgs::msg::Point goal_p =
    transform_point_to_planning_frame(goal_world, goal_frame);

  geometry_msgs::msg::Pose pre_grasp;
  pre_grasp.position = pick_p;
  pre_grasp.position.z += pick_offset_z_;
  pre_grasp.orientation = grasp_q;

  geometry_msgs::msg::Pose grasp = pre_grasp;

  // 保留你当前这套已经验证“Z 基本合适”的逻辑
  const double k_contact_below_top_z = -0.100;
  grasp.position.z = pick_p.z - k_contact_below_top_z;

  RCLCPP_WARN(
    node_->get_logger(),
    "[DEBUG OFFSET] shape=%s | localized=(%.4f, %.4f, %.4f) "
    "| yaw=%.4f rad (%.1f deg) "
    "| local_base=(%.4f, %.4f) | world_offset=(%.4f, %.4f) "
    "| final_pick_world=(%.4f, %.4f, %.4f)",
    shape_type.c_str(),
    localized_obj_world.x,
    localized_obj_world.y,
    localized_obj_world.z,
    object_world_yaw,
    object_world_yaw * 180.0 / M_PI,
    (shape_type == "nought") ? nought_grasp_offset_world_x_ :
    ((shape_type == "cross") ? cross_grasp_offset_world_x_ : 0.0),
    (shape_type == "nought") ? nought_grasp_offset_world_y_ :
    ((shape_type == "cross") ? cross_grasp_offset_world_y_ : 0.0),
    off_x,
    off_y,
    pick_world.x,
    pick_world.y,
    pick_world.z
  );

  RCLCPP_WARN(
    node_->get_logger(),
    "[DEBUG FINAL PICK] shape=%s | nominal_world=(%.4f, %.4f, %.4f) "
    "| localized_world=(%.4f, %.4f, %.4f) "
    "| pick_world=(%.4f, %.4f, %.4f) "
    "| top_z_world=%.4f | yaw=%.4f rad (%.1f deg) "
    "| pre_grasp=(%.4f, %.4f, %.4f) "
    "| grasp=(%.4f, %.4f, %.4f)",
    shape_type.c_str(),
    object_point.point.x,
    object_point.point.y,
    object_point.point.z,
    localized_obj_world.x,
    localized_obj_world.y,
    localized_obj_world.z,
    pick_world.x,
    pick_world.y,
    pick_world.z,
    localized_top_z_world,
    object_world_yaw,
    object_world_yaw * 180.0 / M_PI,
    pre_grasp.position.x,
    pre_grasp.position.y,
    pre_grasp.position.z,
    grasp.position.x,
    grasp.position.y,
    grasp.position.z
  );

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
  ok = ok && move_arm_to_pose_joint(lift);
  // ok = ok && move_arm_linear_to(lift, cartesian_eef_step_, "lift");
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

std::vector<geometry_msgs::msg::PointStamped> cw2::rough_scan(
  geometry_msgs::msg::PointStamped & basket_point)
{
  g_last_task3_objects.clear();

  basket_point.header.frame_id = "";
  basket_point.header.stamp = rclcpp::Time(0);
  basket_point.point.x = 0.0;
  basket_point.point.y = 0.0;
  basket_point.point.z = 0.0;

  const double k_scan_z = 0.60;
  const std::vector<std::pair<double, double>> scan_xy = {
    { 0.50, -0.30}, { 0.50,  0.00}, { 0.50,  0.30}, { 0.25,  0.30},
    { 0.00,  0.30}, {-0.25,  0.30}, {-0.50,  0.30}, {-0.50,  0.00},
    {-0.50, -0.30}, {-0.25, -0.30}, { 0.00, -0.30}, { 0.25, -0.30}
  };

  std::vector<geometry_msgs::msg::Pose> scan_poses;
  scan_poses.reserve(scan_xy.size());
  for (const auto & xy : scan_xy) {
    geometry_msgs::msg::Pose pose;
    pose.position.x = xy.first;
    pose.position.y = xy.second;
    pose.position.z = k_scan_z;
    pose.orientation = make_top_down_q();
    scan_poses.push_back(pose);
  }

  std::vector<pcl::PointXYZRGB> global_world_pts;
  global_world_pts.reserve(400000);

  set_gripper_width(gripper_open_width_);

  for (std::size_t i = 0; i < scan_poses.size(); ++i) {
    const auto & pose = scan_poses[i];
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task3Scan] scan pose %zu -> (%.3f, %.3f, %.2f)",
      i, pose.position.x, pose.position.y, pose.position.z);

    bool moved = false;
    if (i == 0) {
      moved = move_arm_to_pose_joint(pose);
    } else {
      moved = move_arm_linear_to(pose, cartesian_eef_step_, "task3_scan_move");
      if (!moved) {
        moved = move_arm_to_pose_joint(pose);
      }
    }

    if (!moved) {
      RCLCPP_WARN(node_->get_logger(), "[Task3Scan] failed to reach scan pose %zu", i);
      continue;
    }

    rclcpp::sleep_for(std::chrono::milliseconds(500));

    PointCPtr cloud;
    std::string frame_id;
    if (!capture_latest_cloud(cloud, frame_id, 1.5)) {
      RCLCPP_WARN(node_->get_logger(), "[Task3Scan] no cloud at scan pose %zu", i);
      continue;
    }

    std::vector<pcl::PointXYZRGB> world_pts;
    if (!transform_cloud_to_world_points(tf_buffer_, node_->get_logger(), cloud, frame_id, world_pts)) {
      RCLCPP_WARN(node_->get_logger(), "[Task3Scan] transform failed at scan pose %zu", i);
      continue;
    }

    global_world_pts.insert(global_world_pts.end(), world_pts.begin(), world_pts.end());
    RCLCPP_WARN(
      node_->get_logger(),
      "[Task3Scan] accumulated raw world points: +%zu -> %zu",
      world_pts.size(), global_world_pts.size());
  }

  const auto fused_world_pts = downsample_world_points(global_world_pts, 0.003);
  RCLCPP_WARN(
    node_->get_logger(),
    "[Task3Scan] fused world points raw=%zu downsampled=%zu",
    global_world_pts.size(), fused_world_pts.size());

  g_last_task3_objects = detect_task3_objects_from_world_points(node_->get_logger(), fused_world_pts);
  write_task3_debug_outputs(node_->get_logger(), g_last_task3_objects);
  write_task3_fused_cloud_debug_outputs(node_->get_logger(), fused_world_pts, g_last_task3_objects);

  std::vector<geometry_msgs::msg::PointStamped> candidates;
  Task3MergedScanObject best_basket;
  bool have_basket = false;

  for (const auto & obj : g_last_task3_objects) {
    if (obj.category == "basket") {
      if (!have_basket || obj.point_count > best_basket.point_count) {
        best_basket = obj;
        have_basket = true;
      }
      continue;
    }

    if (obj.category != "cross" && obj.category != "nought") {
      continue;
    }

    geometry_msgs::msg::PointStamped p;
    p.header.frame_id = "world";
    p.header.stamp = rclcpp::Time(0);
    p.point = obj.position;
    candidates.push_back(p);
  }

  if (have_basket) {
    basket_point.header.frame_id = "world";
    basket_point.header.stamp = rclcpp::Time(0);
    basket_point.point = best_basket.position;
    basket_point.point.z = best_basket.top_z;
  }

  return candidates;
}

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

  int n_cross = 0;
  int n_nought = 0;
  for (const auto & obj : g_last_task3_objects) {
    if (obj.category == "cross") {
      ++n_cross;
    } else if (obj.category == "nought") {
      ++n_nought;
    }
  }

  response->total_num_shapes = static_cast<std::int64_t>(n_cross + n_nought);
  response->num_most_common_shape = static_cast<std::int64_t>(std::max(n_cross, n_nought));

  if (response->total_num_shapes <= 0) {
    RCLCPP_WARN(node_->get_logger(), "[Task3] no valid shapes found");
    return;
  }

  if (basket_point.header.frame_id.empty()) {
    RCLCPP_WARN(node_->get_logger(), "[Task3] basket missing; counting only");
    return;
  }

  int best_idx = -1;
  std::string best_shape_type;
  double best_dist = std::numeric_limits<double>::infinity();

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto & p = candidates[i];

    const auto it = std::find_if(
      g_last_task3_objects.begin(), g_last_task3_objects.end(),
      [&p](const Task3MergedScanObject & obj) {
        return (obj.category == "cross" || obj.category == "nought") &&
               std::hypot(obj.position.x - p.point.x, obj.position.y - p.point.y) < 1e-4;
      });

    if (it == g_last_task3_objects.end()) {
      continue;
    }

    const double dist_to_basket = std::hypot(
      p.point.x - basket_point.point.x,
      p.point.y - basket_point.point.y);

    if (dist_to_basket < best_dist) {
      best_dist = dist_to_basket;
      best_idx = static_cast<int>(i);
      best_shape_type = it->category;
    }
  }

  if (best_idx < 0) {
    RCLCPP_WARN(node_->get_logger(), "[Task3] no valid target near basket found");
    return;
  }

  geometry_msgs::msg::PointStamped place_point = basket_point;
  place_point.point.x += (basket_point.point.x >= 0.0) ? -0.02 : 0.02;
  place_point.point.y += (basket_point.point.y >= 0.0) ? -0.02 : 0.02;

  RCLCPP_INFO(
    node_->get_logger(),
    "[Task3] total=%lld cross=%d nought=%d picked_shape=%s target_idx=%d dist_to_basket=%.3f",
    static_cast<long long>(response->total_num_shapes),
    n_cross, n_nought, best_shape_type.c_str(), best_idx, best_dist);

  pick_and_place(candidates[static_cast<std::size_t>(best_idx)], place_point, best_shape_type);
}