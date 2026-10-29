// Runs the pick-and-place policy at 20 Hz: /joint_states + /object_pose + TF -> joint position targets.
// dry_run (default true) only logs the targets; nothing is published to the robot.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "so101_policy_bridge/policy_core.hpp"

using namespace std::chrono_literals;
using so101_policy_bridge::Inputs;
using so101_policy_bridge::Joints;
using so101_policy_bridge::PolicyCore;

namespace
{
Joints to_joints(const std::vector<double> & v, const std::string & name)
{
  if (v.size() != 6) {
    throw std::runtime_error("parameter " + name + " needs 6 values");
  }
  Joints j;
  std::copy(v.begin(), v.end(), j.begin());
  return j;
}
}  // namespace

class BridgeNode : public rclcpp::Node
{
public:
  BridgeNode()
  : Node("so101_policy_bridge"), tf_buffer_(get_clock()), tf_listener_(tf_buffer_)
  {
    dry_run_ = declare_parameter("dry_run", true);
    joint_names_ = declare_parameter("joint_names", std::vector<std::string>{});
    max_delta_ = to_joints(declare_parameter("max_delta", std::vector<double>{}), "max_delta");
    lower_ = to_joints(declare_parameter("joint_lower", std::vector<double>{}), "joint_lower");
    upper_ = to_joints(declare_parameter("joint_upper", std::vector<double>{}), "joint_upper");
    ws_min_ = declare_parameter("workspace_min", std::vector<double>{});
    ws_max_ = declare_parameter("workspace_max", std::vector<double>{});
    joint_timeout_ = declare_parameter("joint_state_timeout", 0.1);
    object_timeout_ = declare_parameter("object_pose_timeout", 0.2);
    base_frame_ = declare_parameter("base_frame", std::string("base_link"));
    integrate_targets_ = declare_parameter("integrate_targets", true);
    max_lead_ = declare_parameter("max_lead", 0.15);
    auto zone = declare_parameter("zone_pos", std::vector<double>{});
    const auto model = declare_parameter("model_path", std::string());
    const double rate = declare_parameter("rate_hz", 20.0);
    if (joint_names_.size() != 6 || zone.size() != 3 || ws_min_.size() != 3 || ws_max_.size() != 3) {
      throw std::runtime_error("check joint_names (6), zone_pos (3), workspace_min/max (3)");
    }

    core_ = std::make_unique<PolicyCore>(model, std::array<double, 3>{zone[0], zone[1], zone[2]});
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10, [this](sensor_msgs::msg::JointState::SharedPtr m) { on_joints(*m); });
    object_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "/object_pose", 10, [this](geometry_msgs::msg::PoseStamped::SharedPtr m) { on_object(*m); });
    cmd_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
      declare_parameter("command_topic", std::string("/forward_position_controller/commands")), 10);
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this] { tick(); });
    RCLCPP_INFO(get_logger(), "policy %s loaded, %s", model.c_str(),
      dry_run_ ? "DRY RUN: commands are only logged" : "LIVE: publishing joint targets");
  }

private:
  void on_joints(const sensor_msgs::msg::JointState & m)
  {
    // Messages may list joints in any order; pick ours by name.
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      auto it = std::find(m.name.begin(), m.name.end(), joint_names_[i]);
      if (it == m.name.end()) {
        return;
      }
      const size_t k = it - m.name.begin();
      in_.qpos[i] = m.position[k];
      in_.qvel[i] = k < m.velocity.size() ? m.velocity[k] : 0.0;
    }
    last_joints_ = now();
    have_joints_ = true;
  }

  void on_object(const geometry_msgs::msg::PoseStamped & m)
  {
    if (m.header.frame_id != base_frame_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "object pose must be in %s, got %s",
        base_frame_.c_str(), m.header.frame_id.c_str());
      return;
    }
    in_.cube_pos = {m.pose.position.x, m.pose.position.y, m.pose.position.z};
    const auto & q = m.pose.orientation;
    in_.cube_yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    last_object_ = now();
    have_object_ = true;
  }

  bool tcp_from_tf()
  {
    // TCP = midpoint of the two jaw tips, oriented like gripper_link (as in the sim).
    try {
      auto a = tf_buffer_.lookupTransform(base_frame_, "Fixed_Jaw_tip", tf2::TimePointZero);
      auto b = tf_buffer_.lookupTransform(base_frame_, "Moving_Jaw_tip", tf2::TimePointZero);
      auto g = tf_buffer_.lookupTransform(base_frame_, "gripper_link", tf2::TimePointZero);
      const auto & ta = a.transform.translation;
      const auto & tb = b.transform.translation;
      const auto & r = g.transform.rotation;
      in_.tcp_pose = {(ta.x + tb.x) / 2, (ta.y + tb.y) / 2, (ta.z + tb.z) / 2, r.w, r.x, r.y, r.z};
      return true;
    } catch (const tf2::TransformException & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "TF not ready: %s", e.what());
      return false;
    }
  }

  void stop(const char * why)
  {
    // Stale input: send nothing (servos hold their last target) and start a fresh episode later.
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "holding: %s", why);
    core_->reset();
    have_target_ = false;
  }

  void tick()
  {
    const auto t = now();
    if (!have_joints_ || (t - last_joints_).seconds() > joint_timeout_) {
      return stop("joint_states missing or older than joint_state_timeout");
    }
    if (!have_object_ || (t - last_object_).seconds() > object_timeout_) {
      return stop("object_pose missing or older than object_pose_timeout");
    }
    if (!tcp_from_tf()) {
      return stop("no TF for the gripper");
    }
    for (int i = 0; i < 3; ++i) {
      if (in_.tcp_pose[i] < ws_min_[i] || in_.tcp_pose[i] > ws_max_[i]) {
        return stop("gripper outside the workspace box");
      }
    }

    const Joints action = core_->step(in_);
    // The sim's joints (stiffness 1000) reach every target, so "measured + step" there equals "last target
    // + step". The real servos (~17 N*m/rad) sag under load: stepping from the measured pose let the sag add
    // up every step (first live run: the gripper sank into the table). Stepping from the last target keeps
    // the sag a constant offset; max_lead caps how far the target may run ahead of the measured pose.
    Joints base = in_.qpos;
    if (integrate_targets_ && have_target_) {
      for (size_t i = 0; i < base.size(); ++i) {
        base[i] = std::clamp(last_target_[i], in_.qpos[i] - max_lead_, in_.qpos[i] + max_lead_);
      }
    }
    Joints target = PolicyCore::targets(base, action, max_delta_);
    for (size_t i = 0; i < target.size(); ++i) {
      target[i] = std::clamp(target[i], lower_[i], upper_[i]);
    }
    last_target_ = target;
    have_target_ = true;

    std_msgs::msg::Float64MultiArray msg;
    msg.data.assign(target.begin(), target.end());
    if (dry_run_) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
        "[dry run] action [%.2f %.2f %.2f %.2f %.2f %.2f] -> target [%.3f %.3f %.3f %.3f %.3f %.3f]",
        action[0], action[1], action[2], action[3], action[4], action[5],
        target[0], target[1], target[2], target[3], target[4], target[5]);
    } else {
      cmd_pub_->publish(msg);
    }
  }

  bool dry_run_;
  std::vector<std::string> joint_names_;
  Joints max_delta_, lower_, upper_;
  std::vector<double> ws_min_, ws_max_;
  double joint_timeout_, object_timeout_;
  std::string base_frame_;
  bool integrate_targets_;
  double max_lead_;
  Joints last_target_{};
  bool have_target_ = false;

  std::unique_ptr<PolicyCore> core_;
  Inputs in_;
  bool have_joints_ = false, have_object_ = false;
  rclcpp::Time last_joints_, last_object_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr object_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BridgeNode>());
  rclcpp::shutdown();
  return 0;
}
