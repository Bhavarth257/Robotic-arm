// Replays sim episodes (recorded by scripts/record_replay.py) through PolicyCore: the C++ observation
// must match the sim's, and the C++ action must match the sim policy's action, at every step.
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "so101_policy_bridge/policy_core.hpp"

using so101_policy_bridge::Inputs;
using so101_policy_bridge::Joints;
using so101_policy_bridge::PolicyCore;

namespace
{
std::vector<std::vector<double>> read_csv(const std::string & path)
{
  std::ifstream f(path);
  std::string line;
  std::getline(f, line);  // header
  std::vector<std::vector<double>> rows;
  while (std::getline(f, line)) {
    std::vector<double> r;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) {
      r.push_back(std::stod(cell));
    }
    rows.push_back(r);
  }
  return rows;
}
}  // namespace

TEST(PolicyCore, ReplayMatchesSim)
{
  const auto rows = read_csv(TEST_DATA_DIR "/replay.csv");
  ASSERT_GT(rows.size(), 100u);
  PolicyCore core(MODEL_PATH, {0.15, 0.20, 0.0005});
  const size_t n = core.n_obs();

  double worst_obs = 0.0, worst_act = 0.0;
  for (const auto & r : rows) {
    // Columns: start, qpos[6], qvel[6], tcp[7], cube xyz[3], cube yaw, obs[n], action[6].
    if (r[0] > 0.5) {
      core.reset();
    }
    Inputs in;
    size_t c = 1;
    for (auto & v : in.qpos) v = r[c++];
    for (auto & v : in.qvel) v = r[c++];
    for (auto & v : in.tcp_pose) v = r[c++];
    for (auto & v : in.cube_pos) v = r[c++];
    in.cube_yaw = r[c++];

    const auto obs = core.build_obs(in);
    for (size_t i = 0; i < n; ++i) {
      worst_obs = std::max(worst_obs, std::abs(obs[i] - r[c + i]));
    }
    const Joints a = core.step(in);
    for (size_t i = 0; i < 6; ++i) {
      const double expected = std::max(-1.0, std::min(1.0, r[c + n + i]));
      worst_act = std::max(worst_act, std::abs(a[i] - expected));
    }
  }
  // 1e-4, like the M6 ONNX parity limit: inputs agree to ~1e-7, but prev_action feeds back the
  // policy output, which differs from the float32 sim by ~5e-5 (double vs float input rounding).
  EXPECT_LT(worst_obs, 1e-4) << "observation differs from the sim";
  EXPECT_LT(worst_act, 1e-4) << "action differs from the sim policy";
  std::printf("replayed %zu steps: max obs diff %.2e, max action diff %.2e\n", rows.size(), worst_obs, worst_act);
}

TEST(PolicyCore, TargetsAddScaledAction)
{
  const Joints q{0, 0.1, -0.2, 0.3, 0, 0.6};
  const Joints a{1, -1, 0.5, 0, 0, -1};
  const Joints d{0.05, 0.05, 0.05, 0.05, 0.05, 0.2};
  const Joints t = PolicyCore::targets(q, a, d);
  EXPECT_NEAR(t[0], 0.05, 1e-12);
  EXPECT_NEAR(t[1], 0.05, 1e-12);
  EXPECT_NEAR(t[2], -0.175, 1e-12);
  EXPECT_NEAR(t[5], 0.4, 1e-12);
}
