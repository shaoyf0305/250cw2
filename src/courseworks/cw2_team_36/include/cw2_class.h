/* feel free to change any part of this file, or delete this file. In general,
you can do whatever you want with this template code, including deleting it all
and starting from scratch. The only requirment is to make sure your entire
solution is contained within the cw2_team_<your_team_number> package */

#ifndef CW2_CLASS_H_
#define CW2_CLASS_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "cw2_world_spawner/srv/task1_service.hpp"
#include "cw2_world_spawner/srv/task2_service.hpp"
#include "cw2_world_spawner/srv/task3_service.hpp"

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointC;
typedef PointC::Ptr PointCPtr;

class cw2
{
public:
  explicit cw2(const rclcpp::Node::SharedPtr &node);

  void t1_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task1Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task1Service::Response> response);
  void t2_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task2Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task2Service::Response> response);
  void t3_callback(
    const std::shared_ptr<cw2_world_spawner::srv::Task3Service::Request> request,
    std::shared_ptr<cw2_world_spawner::srv::Task3Service::Response> response);

  void cloud_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg);

  // Task 1 pick-and-place (same motion pattern as comp0250 CW1).
  geometry_msgs::msg::Quaternion make_top_down_q() const;
  geometry_msgs::msg::Quaternion ee_orientation_for_shape(const std::string & shape) const;
  bool move_arm_to_pose_joint(const geometry_msgs::msg::Pose & target_pose);
  bool move_arm_linear_to(
    const geometry_msgs::msg::Pose & target_pose,
    double eef_step,
    const char * step_name = nullptr);  bool set_gripper_width(double width_m);
  geometry_msgs::msg::Point transform_point_to_planning_frame(
    const geometry_msgs::msg::Point & p, const std::string & source_frame);
  geometry_msgs::msg::Quaternion ee_orientation_for_shape_with_world_yaw(
    const std::string & shape, double world_yaw_rad) const;

  bool capture_latest_cloud(PointCPtr & cloud, std::string & frame_id, double timeout_sec = 2.0);

  bool estimate_object_yaw_from_cloud(
    const PointCPtr & cloud,
    const std::string & cloud_frame_id,
    const geometry_msgs::msg::PointStamped & object_loc,
    double & estimated_yaw_rad) const;

  // Task 2: move above object centroid, classify nought vs cross from depth cloud.
  bool move_to_task2_observation_pose(const geometry_msgs::msg::PointStamped & object_loc);
  std::string classify_shape_from_cloud(
    const PointCPtr & cloud, const std::string & cloud_frame_id,
    const geometry_msgs::msg::PointStamped & object_loc);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Service<cw2_world_spawner::srv::Task1Service>::SharedPtr t1_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task2Service>::SharedPtr t2_service_;
  rclcpp::Service<cw2_world_spawner::srv::Task3Service>::SharedPtr t3_service_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr color_cloud_sub_;
  rclcpp::CallbackGroup::SharedPtr pointcloud_callback_group_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> hand_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex cloud_mutex_;
  PointCPtr g_cloud_ptr;
  std::uint64_t g_cloud_sequence_ = 0;
  std::string g_input_pc_frame_id_;

  std::string pointcloud_topic_;
  bool pointcloud_qos_reliable_ = false;

  std::string deep_scan(const geometry_msgs::msg::PointStamped & object_loc);
  bool pick_and_place(
  const geometry_msgs::msg::PointStamped & object_point,
  const geometry_msgs::msg::PointStamped & goal_point,
  const std::string & shape_type);

std::vector<geometry_msgs::msg::PointStamped> rough_scan(
  geometry_msgs::msg::PointStamped & basket_point);
  
  // Task 1 tuning (defaults match CW1-style pick/place).
  double pick_offset_z_ = 0.26;
  double grasp_descent_z_ = 0.1;
  double place_offset_z_ = 0.35;
  double gripper_open_width_ = 0.09;
  double gripper_grasp_width_ = 0.01;
  double cartesian_eef_step_ = 0.01;
  double cartesian_min_fraction_ = 0.6;

  // Task 1 shape-aware grasp: planar offset from centroid in world XY (metres).
  bool task1_apply_shape_xy_offset_ = true;
  double nought_grasp_offset_world_x_ = 0.075;
  double nought_grasp_offset_world_y_ = 0.0;
  double cross_grasp_offset_world_x_ = 0.042;
  double cross_grasp_offset_world_y_ = 0.0;
  // Extra yaw (rad) about tool z applied for nought only; default +90° to align fingers with ring.
  double nought_grasp_extra_yaw_rad_ = 1.57;

  // Task 2 (shape ID): observation height above centroid (planning frame), inner-hole heuristic.
  double task2_obs_height_above_centroid_m_ = 0.45;
  int task2_settle_ms_ = 600;
  double task2_inner_radius_m_ = 0.05;
  double task2_ring_r_min_m_ = 0.06;
  double task2_ring_r_max_m_ = 0.141;
  double task2_ground_sample_radius_m_ = 0.16;
  double task2_surface_above_ground_m_ = 0.02;
  int task2_inner_min_points_cross_ = 28;
  double task2_ring_inner_ratio_nought_min_ = 3.0;
};

#endif  // CW2_CLASS_H_
