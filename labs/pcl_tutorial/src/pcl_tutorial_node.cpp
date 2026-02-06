/*
 * Student starter node for COMP0250 ROS 2 PCL tutorial.
 * This package compiles, subscribes, and publishes nothing by default.
 * Fill in TODO blocks as part of the lab.
 */

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/sac_segmentation.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointC;
typedef PointC::Ptr PointCPtr;

class PCLTutorial final : public rclcpp::Node
{
public:
  PCLTutorial()
  : rclcpp::Node("pcl_tutorial_node"),
    g_cloud_ptr(new PointC),
    g_cloud_filtered(new PointC),
    g_cloud_filtered2(new PointC),
    g_cloud_plane(new PointC),
    g_cloud_cylinder(new PointC),
    g_tree_ptr(new pcl::search::KdTree<PointT>()),
    g_cloud_normals(new pcl::PointCloud<pcl::Normal>),
    g_cloud_normals2(new pcl::PointCloud<pcl::Normal>),
    g_inliers_plane(new pcl::PointIndices),
    g_inliers_cylinder(new pcl::PointIndices),
    g_coeff_plane(new pcl::ModelCoefficients),
    g_coeff_cylinder(new pcl::ModelCoefficients)
  {
    g_input_topic = this->declare_parameter<std::string>(
      "input_topic", "/r200/camera/depth_registered/points");

    g_output_cloud_topic = this->declare_parameter<std::string>(
      "output_cloud_topic", "/pcl_tutorial/filtered");
    g_output_cyl_pt_topic = this->declare_parameter<std::string>(
      "output_cyl_pt_topic", "/pcl_tutorial/centroid");

    g_enable_voxel = this->declare_parameter<bool>("enable_voxel", true);
    g_leaf_size = this->declare_parameter<double>("leaf_size", 0.01);

    g_enable_pass = this->declare_parameter<bool>("enable_pass", false);
    g_pass_axis = this->declare_parameter<std::string>("pass_axis", "x");
    g_pass_min = this->declare_parameter<double>("pass_min", 0.0);
    g_pass_max = this->declare_parameter<double>("pass_max", 0.7);

    g_enable_outlier = this->declare_parameter<bool>("enable_outlier", false);
    g_outlier_mean_k = this->declare_parameter<int>("outlier_mean_k", 20);
    g_outlier_stddev = this->declare_parameter<double>("outlier_stddev", 1.0);

    g_do_plane = this->declare_parameter<bool>("do_plane", false);
    g_do_cylinder = this->declare_parameter<bool>("do_cylinder", false);

    g_normal_k = this->declare_parameter<int>("normal_k", 50);
    g_plane_normal_dist_weight = this->declare_parameter<double>("plane_normal_dist_weight", 0.1);
    g_plane_max_iterations = this->declare_parameter<int>("plane_max_iterations", 100);
    g_plane_distance = this->declare_parameter<double>("plane_distance", 0.03);

    g_cylinder_normal_dist_weight = this->declare_parameter<double>("cylinder_normal_dist_weight", 0.1);
    g_cylinder_max_iterations = this->declare_parameter<int>("cylinder_max_iterations", 10000);
    g_cylinder_distance = this->declare_parameter<double>("cylinder_distance", 0.05);
    g_cylinder_radius_min = this->declare_parameter<double>("cylinder_radius_min", 0.0);
    g_cylinder_radius_max = this->declare_parameter<double>("cylinder_radius_max", 0.1);

    g_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      g_input_topic, rclcpp::SensorDataQoS(),
      std::bind(&PCLTutorial::cloudCallBackOne, this, std::placeholders::_1));

    g_pub_cloud = this->create_publisher<sensor_msgs::msg::PointCloud2>(g_output_cloud_topic, 1);
    g_pub_cyl_pt = this->create_publisher<geometry_msgs::msg::PointStamped>(g_output_cyl_pt_topic, 1);
  }

private:
  void cloudCallBackOne(const sensor_msgs::msg::PointCloud2::SharedPtr cloud_input_msg)
  {
    g_input_pc_frame_id = cloud_input_msg->header.frame_id;

    // TODO(student-1): Convert incoming ROS PointCloud2 to PCL cloud.

    // TODO(student-2): Build filtering pipeline (voxel -> pass-through -> outlier removal).

    // TODO(student-3): Run optional segmentation when enabled.

    // TODO(student-4): Publish filtered output cloud.

    // Log at most once every 2000 ms to avoid flooding terminal output.
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Student starter node active; TODO blocks are not implemented yet.");
  }

  void applyVX(PointCPtr &in_cloud_ptr, PointCPtr &out_cloud_ptr)
  {
    (void)in_cloud_ptr;
    (void)out_cloud_ptr;
    // TODO(student-5): Implement VoxelGrid downsampling.
  }

  void applyPT(PointCPtr &in_cloud_ptr, PointCPtr &out_cloud_ptr)
  {
    (void)in_cloud_ptr;
    (void)out_cloud_ptr;
    // TODO(student-6): Implement PassThrough filtering.
  }

  void applySOR(PointCPtr &in_cloud_ptr, PointCPtr &out_cloud_ptr)
  {
    (void)in_cloud_ptr;
    (void)out_cloud_ptr;
    // TODO(student-7): Implement StatisticalOutlierRemoval.
  }

  void findNormals(PointCPtr &in_cloud_ptr)
  {
    (void)in_cloud_ptr;
    // TODO(student-8): Implement normal estimation.
  }

  void segPlane(PointCPtr &in_cloud_ptr)
  {
    (void)in_cloud_ptr;
    // TODO(student-9): Implement normal-plane segmentation.
  }

  void segCylind(PointCPtr &in_cloud_ptr)
  {
    (void)in_cloud_ptr;
    // TODO(student-10): Implement cylinder segmentation.
  }

  void findCylPose(PointCPtr &in_cloud_ptr)
  {
    (void)in_cloud_ptr;
    // TODO(student-11): Compute and publish cylinder centroid in target frame.
  }

  void pubFilteredPCMsg(
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pc_pub,
    PointC &pc,
    const std_msgs::msg::Header &header)
  {
    (void)pc_pub;
    (void)pc;
    (void)header;
    // TODO(student-12): Convert PCL cloud back to ROS and publish.
  }

  void publishPose(geometry_msgs::msg::PointStamped &cyl_pt_msg)
  {
    (void)cyl_pt_msg;
    // TODO(student-13): Publish centroid point on output topic.
  }

private:
  std::string g_input_topic;
  std::string g_output_cloud_topic;
  std::string g_output_cyl_pt_topic;

  bool g_enable_voxel;
  double g_leaf_size;

  bool g_enable_pass;
  std::string g_pass_axis;
  double g_pass_min;
  double g_pass_max;

  bool g_enable_outlier;
  int g_outlier_mean_k;
  double g_outlier_stddev;

  bool g_do_plane;
  bool g_do_cylinder;

  int g_normal_k;
  double g_plane_normal_dist_weight;
  int g_plane_max_iterations;
  double g_plane_distance;

  double g_cylinder_normal_dist_weight;
  int g_cylinder_max_iterations;
  double g_cylinder_distance;
  double g_cylinder_radius_min;
  double g_cylinder_radius_max;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr g_sub;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr g_pub_cloud;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr g_pub_cyl_pt;

  tf2_ros::Buffer g_tf_buffer{this->get_clock()};
  tf2_ros::TransformListener g_tf_listener{g_tf_buffer};

  std::string g_input_pc_frame_id;
  pcl::PCLPointCloud2 g_pcl_pc;

  PointCPtr g_cloud_ptr;
  PointCPtr g_cloud_filtered;
  PointCPtr g_cloud_filtered2;
  PointCPtr g_cloud_plane;
  PointCPtr g_cloud_cylinder;

  pcl::VoxelGrid<PointT> g_vx;
  pcl::PassThrough<PointT> g_pt;
  pcl::StatisticalOutlierRemoval<PointT> g_sor;

  pcl::search::KdTree<PointT>::Ptr g_tree_ptr;
  pcl::NormalEstimation<PointT, pcl::Normal> g_ne;
  pcl::PointCloud<pcl::Normal>::Ptr g_cloud_normals;
  pcl::PointCloud<pcl::Normal>::Ptr g_cloud_normals2;

  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> g_seg;
  pcl::ExtractIndices<PointT> g_extract_pc;
  pcl::ExtractIndices<pcl::Normal> g_extract_normals;

  pcl::PointIndices::Ptr g_inliers_plane;
  pcl::PointIndices::Ptr g_inliers_cylinder;
  pcl::ModelCoefficients::Ptr g_coeff_plane;
  pcl::ModelCoefficients::Ptr g_coeff_cylinder;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PCLTutorial>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
