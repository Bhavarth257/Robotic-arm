import gymnasium as gym
import pytest
import torch

import so101_rl.envs  # noqa: F401
from mani_skill.utils.structs.pose import Pose
from so101_rl.envs.pick_place import CUBE_HALF, ZONE_CENTER

N = 16


@pytest.fixture(scope="module")
def env():
    e = gym.make("SO101PickPlace-v1", num_envs=N, obs_mode="state", sim_backend="gpu",
                 control_mode="pd_joint_delta_pos", reward_mode="normalized_dense")
    yield e
    e.close()


def hold(env, steps):
    # Zero joint deltas: arm holds its pose.
    for _ in range(steps):
        out = env.step(torch.zeros(N, 6, device="cuda"))
    return out


def place_cube(env, x, y):
    u = env.unwrapped
    p = torch.tensor([[x, y, CUBE_HALF]] * N, device="cuda")
    u.cube.set_pose(Pose.create_from_pq(p))
    u.cube.set_linear_velocity(torch.zeros(N, 3, device="cuda"))
    # GPU sim: push the new state to PhysX, otherwise the next step overwrites it.
    u.scene._gpu_apply_all()
    u.scene._gpu_fetch_all()


def test_reset_is_deterministic(env):
    obs_a, _ = env.reset(seed=7)
    hold(env, 3)
    obs_b, _ = env.reset(seed=7)
    assert torch.equal(obs_a, obs_b)
    obs_c, _ = env.reset(seed=8)
    assert not torch.equal(obs_a, obs_c)


def test_success_when_cube_in_zone(env):
    env.reset(seed=0)
    place_cube(env, *ZONE_CENTER)
    *_, info = hold(env, 10)
    assert info["success"].all()


@pytest.mark.parametrize("offset", [(0.04, 0.0), (0.0, -0.04), (-0.1, -0.1)])
def test_no_success_outside_zone(env, offset):
    env.reset(seed=0)
    place_cube(env, ZONE_CENTER[0] + offset[0], ZONE_CENTER[1] + offset[1])
    *_, info = hold(env, 10)
    assert not info["success"].any()


def test_reward_finite_for_random_actions_512():
    e = gym.make("SO101PickPlace-v1", num_envs=512, obs_mode="state", sim_backend="gpu",
                 control_mode="pd_joint_delta_pos", reward_mode="normalized_dense")
    e.reset(seed=0)
    torch.manual_seed(0)
    for _ in range(100):
        obs, r, *_ = e.step(torch.rand(512, 6, device="cuda") * 2 - 1)
        assert torch.isfinite(r).all() and torch.isfinite(obs).all()
        assert (r >= -0.1 - 1e-6).all() and (r <= 1 + 1e-6).all()
    e.close()
