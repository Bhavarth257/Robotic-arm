// Overhead-camera ArUco: camera pose from the table marker (id 0, at a measured spot in base_link),
// cube pose from its face markers (id 1 top, 2-5 sides) -> /object_pose in base_link.
// Reads the compressed image stream: raw 1080p images arrive far too slowly over DDS.
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <tf2/LinearMath/Transform.h>
#include <tf2_ros/transform_broadcaster.h>

namespace
{
tf2::Transform to_tf(const cv::Vec3d & rvec, const cv::Vec3d & tvec)
{
  cv::Mat R;
  cv::Rodrigues(rvec, R);
  tf2::Matrix3x3 m(R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2));
  return tf2::Transform(m, tf2::Vector3(tvec[0], tvec[1], tvec[2]));
}

double perimeter(const std::vector<cv::Point2f> & c)
{
  double p = 0;
  for (size_t i = 0; i < c.size(); ++i) {
    p += cv::norm(c[i] - c[(i + 1) % c.size()]);
  }
  return p;
}
}  // namespace

class ArucoNode : public rclcpp::Node
{
public:
  ArucoNode()
  : Node("aruco_cube"), tf_pub_(this)
  {
    base_frame_ = declare_parameter("base_frame", std::string("base_link"));
    table_marker_size_ = declare_parameter("table_marker_size", 0.08);
    cube_marker_size_ = declare_parameter("cube_marker_size", 0.022);
    cube_size_ = declare_parameter("cube_size", 0.03);
    smoothing_ = declare_parameter("camera_pose_smoothing", 0.1);
    // Table marker centre in base_link: x, y, z (m), yaw (rad) of the marker's own x axis.
    const auto tm = declare_parameter("table_marker_pose", std::vector<double>{});
    if (tm.size() != 4) {
      throw std::runtime_error("table_marker_pose needs [x, y, z, yaw]");
    }
    tf2::Quaternion q;
    q.setRPY(0, 0, tm[3]);
    base_from_table_ = tf2::Transform(q, tf2::Vector3(tm[0], tm[1], tm[2]));

    dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    params_ = cv::aruco::DetectorParameters::create();
    params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    const auto qos = rclcpp::SensorDataQoS();
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "camera_info", qos, [this](sensor_msgs::msg::CameraInfo::SharedPtr m) { on_info(*m); });
    image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
      "image_raw/compressed", qos, [this](sensor_msgs::msg::CompressedImage::SharedPtr m) { on_image(*m); });
    pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>("/object_pose", 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::CompressedImage>("~/debug/compressed", 1);
  }

private:
  void on_info(const sensor_msgs::msg::CameraInfo & m)
  {
    if (m.k[0] == 0.0) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "camera_info has no calibration yet");
      return;
    }
    K_ = cv::Mat(3, 3, CV_64F, const_cast<double *>(m.k.data())).clone();
    D_ = cv::Mat(m.d, true).reshape(1, 1);
  }

  void on_image(const sensor_msgs::msg::CompressedImage & m)
  {
    if (K_.empty()) {
      return;
    }
    const cv::Mat img = cv::imdecode(m.data, cv::IMREAD_COLOR);
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(img, dict_, corners, ids, params_);

    // Camera pose from the table marker (kept, smoothed, while the arm hides the marker).
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] == 0) {
        std::vector<cv::Vec3d> rv, tv;
        cv::aruco::estimatePoseSingleMarkers(std::vector<std::vector<cv::Point2f>>{corners[i]},
          table_marker_size_, K_, D_, rv, tv);
        const tf2::Transform base_from_cam = base_from_table_ * to_tf(rv[0], tv[0]).inverse();
        if (!have_cam_) {
          base_from_cam_ = base_from_cam;
          have_cam_ = true;
        } else {
          base_from_cam_.setOrigin(base_from_cam_.getOrigin().lerp(base_from_cam.getOrigin(), smoothing_));
          base_from_cam_.setRotation(base_from_cam_.getRotation().slerp(base_from_cam.getRotation(), smoothing_));
        }
      }
    }
    if (!have_cam_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "table marker (id 0) not seen yet");
      publish_debug(img, ids, corners, m.header);
      return;
    }
    broadcast_camera(m.header);

    // Cube: prefer the top marker, else the largest side marker in view.
    int best = -1;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (ids[i] < 1 || ids[i] > 5) {
        continue;
      }
      if (best < 0 || ids[i] == 1 ||
        (ids[best] != 1 && perimeter(corners[i]) > perimeter(corners[best])))
      {
        best = static_cast<int>(i);
      }
    }
    if (best >= 0) {
      std::vector<cv::Vec3d> rv, tv;
      cv::aruco::estimatePoseSingleMarkers(std::vector<std::vector<cv::Point2f>>{corners[best]},
        cube_marker_size_, K_, D_, rv, tv);
      const tf2::Transform base_from_marker = base_from_cam_ * to_tf(rv[0], tv[0]);
      // Marker z points out of the face, so the centre is half a cube behind it.
      const tf2::Vector3 centre = base_from_marker * tf2::Vector3(0, 0, -cube_size_ / 2);
      const tf2::Matrix3x3 & R = base_from_marker.getBasis();
      // Top face: yaw of the marker's x axis. Side face: yaw of its outward normal.
      // Both give the cube's yaw up to a multiple of 90 deg, which is all the policy uses.
      const double yaw = ids[best] == 1 ? std::atan2(R[1][0], R[0][0]) : std::atan2(R[1][2], R[0][2]);
      tf2::Quaternion q;
      q.setRPY(0, 0, yaw);

      geometry_msgs::msg::PoseStamped p;
      p.header.stamp = m.header.stamp;
      p.header.frame_id = base_frame_;
      p.pose.position.x = centre.x();
      p.pose.position.y = centre.y();
      p.pose.position.z = centre.z();
      p.pose.orientation.x = q.x();
      p.pose.orientation.y = q.y();
      p.pose.orientation.z = q.z();
      p.pose.orientation.w = q.w();
      pose_pub_->publish(p);
    }
    publish_debug(img, ids, corners, m.header);
  }

  void broadcast_camera(const std_msgs::msg::Header & h)
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = h.stamp;
    t.header.frame_id = base_frame_;
    t.child_frame_id = h.frame_id;
    const auto & o = base_from_cam_.getOrigin();
    const auto r = base_from_cam_.getRotation();
    t.transform.translation.x = o.x();
    t.transform.translation.y = o.y();
    t.transform.translation.z = o.z();
    t.transform.rotation.x = r.x();
    t.transform.rotation.y = r.y();
    t.transform.rotation.z = r.z();
    t.transform.rotation.w = r.w();
    tf_pub_.sendTransform(t);
  }

  void publish_debug(cv::Mat img, const std::vector<int> & ids,
    const std::vector<std::vector<cv::Point2f>> & corners, const std_msgs::msg::Header & h)
  {
    if (debug_pub_->get_subscription_count() == 0) {
      return;  // drawing + JPEG encoding at 30 Hz is only worth it when someone watches
    }
    cv::aruco::drawDetectedMarkers(img, corners, ids);
    cv::resize(img, img, cv::Size(), 0.5, 0.5);
    sensor_msgs::msg::CompressedImage out;
    out.header = h;
    out.format = "jpeg";
    cv::imencode(".jpg", img, out.data);
    debug_pub_->publish(out);
  }

  std::string base_frame_;
  double table_marker_size_, cube_marker_size_, cube_size_, smoothing_;
  tf2::Transform base_from_table_, base_from_cam_;
  bool have_cam_ = false;
  cv::Mat K_, D_;
  cv::Ptr<cv::aruco::Dictionary> dict_;
  cv::Ptr<cv::aruco::DetectorParameters> params_;
  tf2_ros::TransformBroadcaster tf_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr debug_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ArucoNode>());
  rclcpp::shutdown();
  return 0;
}
