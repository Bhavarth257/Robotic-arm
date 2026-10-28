// Builds the policy observation exactly as SO101PickPlace-v1 does and runs the exported ONNX actor.
// No ROS here, so it can be unit-tested against recorded sim episodes.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace so101_policy_bridge
{

using Joints = std::array<double, 6>;  // shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper

struct Inputs
{
  Joints qpos{};                    // rad
  Joints qvel{};                    // rad/s
  std::array<double, 7> tcp_pose{}; // base frame: x, y, z, qw, qx, qy, qz (ManiSkill order)
  std::array<double, 3> cube_pos{}; // base frame, m
  double cube_yaw = 0.0;            // rad, about base z
};

class PolicyCore
{
public:
  PolicyCore(const std::string & model_path, const std::array<double, 3> & zone_pos);

  // Forget previous actions; call at the start of every episode.
  void reset();

  std::vector<float> build_obs(const Inputs & in) const;

  // Runs the policy and returns its action clipped to [-1, 1]; remembers it for the next obs.
  Joints step(const Inputs & in);

  // Joint position targets for pd_joint_delta_pos: current position + action * max step.
  static Joints targets(const Joints & qpos, const Joints & action, const Joints & max_delta);

  size_t n_obs() const { return n_obs_; }

private:
  std::array<double, 3> zone_pos_;
  Joints prev_action_{};
  Joints prev_action2_{};
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  size_t n_obs_ = 0;
};

}  // namespace so101_policy_bridge
