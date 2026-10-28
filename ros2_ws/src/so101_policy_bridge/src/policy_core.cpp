#include "so101_policy_bridge/policy_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace so101_policy_bridge
{

PolicyCore::PolicyCore(const std::string & model_path, const std::array<double, 3> & zone_pos)
: zone_pos_(zone_pos), env_(ORT_LOGGING_LEVEL_WARNING, "so101_policy")
{
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);  // tiny MLP: threads only add overhead
  session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
  n_obs_ = static_cast<size_t>(session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1]);
  if (n_obs_ != 45) {
    throw std::runtime_error("expected a 45-input policy, model has " + std::to_string(n_obs_));
  }
}

void PolicyCore::reset()
{
  prev_action_.fill(0.0);
  prev_action2_.fill(0.0);
}

std::vector<float> PolicyCore::build_obs(const Inputs & in) const
{
  // Same order as the env's flattened state_dict: agent (qpos, qvel), then extra in insertion order.
  std::vector<float> o;
  o.reserve(n_obs_);
  auto put = [&o](double v) { o.push_back(static_cast<float>(v)); };

  for (double v : in.qpos) put(v);
  for (double v : in.qvel) put(v);

  // q and -q are the same rotation; the sim always reports w >= 0.
  const double s = in.tcp_pose[3] < 0.0 ? -1.0 : 1.0;
  for (int i = 0; i < 3; ++i) put(in.tcp_pose[i]);
  for (int i = 3; i < 7; ++i) put(s * in.tcp_pose[i]);

  for (double v : in.cube_pos) put(v);
  put(std::sin(4.0 * in.cube_yaw));  // a cube looks the same every 90 deg
  put(std::cos(4.0 * in.cube_yaw));
  for (double v : zone_pos_) put(v);
  for (int i = 0; i < 3; ++i) put(in.cube_pos[i] - in.tcp_pose[i]);
  for (int i = 0; i < 3; ++i) put(zone_pos_[i] - in.cube_pos[i]);
  for (double v : prev_action_) put(v);
  for (double v : prev_action2_) put(v);
  return o;
}

Joints PolicyCore::step(const Inputs & in)
{
  std::vector<float> obs = build_obs(in);
  const std::array<int64_t, 2> shape{1, static_cast<int64_t>(n_obs_)};
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input = Ort::Value::CreateTensor<float>(mem, obs.data(), obs.size(), shape.data(), shape.size());
  const char * in_names[] = {"obs"};
  const char * out_names[] = {"action"};
  auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
  const float * a = out[0].GetTensorData<float>();

  Joints action;
  for (size_t i = 0; i < action.size(); ++i) {
    action[i] = std::clamp(static_cast<double>(a[i]), -1.0, 1.0);
  }
  prev_action2_ = prev_action_;
  prev_action_ = action;
  return action;
}

Joints PolicyCore::targets(const Joints & qpos, const Joints & action, const Joints & max_delta)
{
  Joints t;
  for (size_t i = 0; i < t.size(); ++i) {
    t[i] = qpos[i] + action[i] * max_delta[i];
  }
  return t;
}

}  // namespace so101_policy_bridge
